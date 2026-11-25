"""Step 3 (RQ1): Bayesian non-linear mixed-effects exponential-plateau model of amblyopic-eye VA (NumPyro NUTS).
  VA_ij = V0_i - G_i * (1 - exp(-t_ij / tau_i)) + e_ij,   e ~ Student-t(4, 0, sigma_chart)
  V0_i      ~ N(muV + uV_trial, sV)
  G_i       = muG + bG_tx + bGV*(V0_i-0.6) + bGA*age_c + uG_trial + sG*zG_i
  log tau_i = muT + bT_tx + bTV*(V0_i-0.6) + bTA*age_c + uT_trial + sT*zT_i,   corr(zG, zT) ~ LKJ(2)
Fold = held-out main trial (ATS5/ATS13/ATS15/ATS16) or ALL. Training = all main + aux trials except the fold.
Held-out evaluation: for each patient, condition on the first 2 observations (importance sampling over the population
predictive with a NEW trial effect), predict later observations; report |error| and 90% interval coverage.
usage: python fit_trajectory_model.py <root> <fold> [warmup samples chains]   |   python fit_trajectory_model.py <root> SUMMARY
Set NOAUX=1 to fit the main trials only (sensitivity analysis).
"""
import os, sys, json, time
ROOT = sys.argv[1].rstrip("/") + "/"; FOLD = sys.argv[2]
NW, NS, NC = (int(x) for x in (sys.argv[3:6] if len(sys.argv) > 5 else (1000, 1000, 4)))
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={NC}"
import numpy as np, pandas as pd
OUT = ROOT + "results/trajectory_model/"; os.makedirs(OUT, exist_ok=True)
NOAUX = os.environ.get("NOAUX") == "1"; PFX = "noaux_" if NOAUX else ""   # sensitivity analysis: main trials only
MAIN = ["ATS5", "ATS13", "ATS15", "ATS16"]
TXS = ["patching", "atropine"]            # spectacles = reference
CHARTS = ["hotv", "snellen", "etdrs", "mixed"]

def load():
    T = pd.read_csv(ROOT + "derived/traj.csv"); P = pd.read_csv(ROOT + "derived/pts.csv")
    P = P[P.role.isin(["main"] if NOAUX else ["main", "aux"])].copy(); T = T.merge(P[["trial", "pid"]], on=["trial", "pid"])
    P["age_c"] = (P.age.fillna(P.age.median()) - 5.5) / 2.0
    T["chart"] = T.chart.fillna("mixed").where(T.chart.isin(CHARTS), "mixed")
    return T, P

def summary():
    rows = [json.load(open(OUT + f"fold_{f}.json")) for f in MAIN if os.path.exists(OUT + f"fold_{f}.json")]
    L = ["RQ1: leave-one-trial-out prediction of later visits from the first 2 observations", ""]
    allerr, allcov = [], []
    for r in rows:
        L.append(f"{r['fold']:6s} n_pat {r['n_pat']:4d} n_pred {r['n_pred']:5d} | |err| median {r['med_abs_err']:.3f} mean {r['mean_abs_err']:.3f} "
                 f"| 90% PI coverage {r['cov90']:.3f} | share |err|<=0.1 {r['share_le_0.1']:.2f} | max R-hat {r['rhat_max']:.3f} | {r['minutes']:.0f} min")
        e = np.load(OUT + f"pred_{r['fold']}.npz"); allerr.append(e["err"]); allcov.append(e["cov"])
    if rows:
        err, cov = np.concatenate(allerr), np.concatenate(allcov)
        a, b = np.median(np.abs(err)), cov.mean()
        L += ["", f"pooled: |err| median {a:.3f}, 90% PI coverage {b:.3f}",
              f"RQ1 criterion (median |err| <= 0.1 AND coverage in [0.85, 0.95]): {'MET' if (a <= 0.1 + 1e-9 and 0.85 <= b <= 0.95) else 'NOT MET'}"]
    if os.path.exists(OUT + "fold_ALL.json"):
        r = json.load(open(OUT + "fold_ALL.json")); L += ["", "Full-data fit (all main + aux trials):", r["params"], f"max R-hat {r['rhat_max']:.3f}"]
    print("\n".join(L)); open(OUT + "summary.txt", "w").write("\n".join(L))

if FOLD == "SUMMARY":
    summary(); sys.exit()

import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
numpyro.set_host_device_count(NC)
T, P = load()
tr = P[P.trial != FOLD] if FOLD != "ALL" else P
trials = sorted(tr.trial.unique()); tix = {t: i for i, t in enumerate(trials)}
key = list(zip(tr.trial, tr.pid)); pix = {k: i for i, k in enumerate(key)}
Ttr = T[[(a, b) in pix for a, b in zip(T.trial, T.pid)]]
obs_p = np.array([pix[(a, b)] for a, b in zip(Ttr.trial, Ttr.pid)])
X = dict(trial=jnp.array([tix[t] for t in tr.trial]), tx=jnp.array(np.column_stack([(tr.tx == t).values for t in TXS]).astype(float)),
         age=jnp.array(tr.age_c.values), p=jnp.array(obs_p), t=jnp.array(Ttr.wk.values), ch=jnp.array([CHARTS.index(c) for c in Ttr.chart]),
         y=jnp.array(Ttr.va.values))
NP, NT = len(tr), len(trials)

def model(X, y=None):
    muV = numpyro.sample("muV", dist.Normal(0.6, 0.3)); sV = numpyro.sample("sV", dist.HalfNormal(0.3))
    muG = numpyro.sample("muG", dist.Normal(0.2, 0.2)); muT = numpyro.sample("muT", dist.Normal(np.log(8.), 1.0))
    bG = numpyro.sample("bG", dist.Normal(0, 0.2).expand([2])); bT = numpyro.sample("bT", dist.Normal(0, 0.5).expand([2]))
    bGV = numpyro.sample("bGV", dist.Normal(0, 0.5)); bTV = numpyro.sample("bTV", dist.Normal(0, 1.0))
    bGA = numpyro.sample("bGA", dist.Normal(0, 0.2)); bTA = numpyro.sample("bTA", dist.Normal(0, 0.5))
    tV, tG, tT = (numpyro.sample(n, dist.HalfNormal(s)) for n, s in (("tV", 0.1), ("tG", 0.1), ("tT", 0.3)))
    with numpyro.plate("trials", NT):
        uV = numpyro.sample("uV_z", dist.Normal(0, 1)) * tV
        uG = numpyro.sample("uG_z", dist.Normal(0, 1)) * tG
        uT = numpyro.sample("uT_z", dist.Normal(0, 1)) * tT
    sG = numpyro.sample("sG", dist.HalfNormal(0.2)); sT = numpyro.sample("sT", dist.HalfNormal(0.7))
    Lc = numpyro.sample("Lc", dist.LKJCholesky(2, 2.0))
    sig = numpyro.sample("sig", dist.HalfNormal(0.1).expand([len(CHARTS)]))
    with numpyro.plate("pts", NP):
        zV = numpyro.sample("zV", dist.Normal(0, 1)); z = numpyro.sample("z", dist.Normal(0, 1).expand([2]).to_event(1))
    zc = z @ Lc.T
    V0 = muV + uV[X["trial"]] + sV * zV; vc = V0 - 0.6
    G = muG + X["tx"] @ bG + bGV * vc + bGA * X["age"] + uG[X["trial"]] + sG * zc[:, 0]
    lt = muT + X["tx"] @ bT + bTV * vc + bTA * X["age"] + uT[X["trial"]] + sT * zc[:, 1]
    p = X["p"]; mu = V0[p] - G[p] * (1 - jnp.exp(-X["t"] / jnp.exp(lt[p])))
    numpyro.sample("y", dist.StudentT(4.0, mu, sig[X["ch"]]), obs=y)

t0 = time.time()
mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=NW, num_samples=NS, num_chains=NC, chain_method="parallel", progress_bar=False)
mcmc.run(jax.random.PRNGKey(2026), X, y=X["y"])
from numpyro.diagnostics import summary as nsum
hyp = ["muV", "sV", "muG", "muT", "bG", "bT", "bGV", "bTV", "bGA", "bTA", "tV", "tG", "tT", "sG", "sT", "sig"]
S = mcmc.get_samples(group_by_chain=True)
sm = nsum({k: S[k] for k in hyp + ["Lc"]})
rhat = float(max(np.nanmax(np.asarray(v["r_hat"])) for v in sm.values()))
D = {k: np.asarray(v).reshape(-1, *np.asarray(v).shape[2:]) for k, v in S.items() if k in hyp + ["Lc", "uV_z", "uG_z", "uT_z"]}
D["rho"] = D["Lc"][:, 1, 0]
np.savez_compressed(OUT + f"draws_{PFX}{FOLD}.npz", trials=np.array(trials), **D)
par = "; ".join(f"{k} {np.median(D[k], 0).round(3).tolist()}" for k in hyp + ["rho"])
res = dict(fold=FOLD, rhat_max=rhat, minutes=(time.time() - t0) / 60, params=par, n_train_pat=int(NP))

if FOLD != "ALL":
    te = P[P.trial == FOLD]; Tte = T[T.trial == FOLD]
    rng = np.random.default_rng(1); M = 4000; sidx = rng.choice(len(D["muG"]), M)
    d = {k: D[k][sidx] for k in hyp + ["rho"]}
    nu = 4.0; errs, covs = [], []
    for (pid, g_), prow in zip(Tte.groupby("pid"), [None] * Tte.pid.nunique()):
        row = te[te.pid == pid].iloc[0]; t = g_.wk.values; y = g_.va.values; ch = np.array([CHARTS.index(c) if c in CHARTS else 3 for c in g_.chart])
        if len(t) < 3: continue
        tx = np.array([row.tx == x for x in TXS], float)
        uV = rng.standard_normal(M) * d["tV"]; uG = rng.standard_normal(M) * d["tG"]; uT = rng.standard_normal(M) * d["tT"]
        V0 = d["muV"] + uV + d["sV"] * rng.standard_normal(M); vc = V0 - 0.6
        z1, z2 = rng.standard_normal(M), rng.standard_normal(M); zg = z1; zt = d["rho"] * z1 + np.sqrt(1 - d["rho"] ** 2) * z2
        G = d["muG"] + d["bG"] @ tx + d["bGV"] * vc + d["bGA"] * row.age_c + uG + d["sG"] * zg
        lt = d["muT"] + d["bT"] @ tx + d["bTV"] * vc + d["bTA"] * row.age_c + uT + d["sT"] * zt
        mu = V0[:, None] - G[:, None] * (1 - np.exp(-t[None, :] / np.exp(lt)[:, None]))
        s = d["sig"][:, ch]
        r = (y[None, :2] - mu[:, :2]) / s[:, :2]
        lw = (-(nu + 1) / 2 * np.log1p(r ** 2 / nu) - np.log(s[:, :2])).sum(1); w = np.exp(lw - lw.max()); w /= w.sum()
        for j in range(2, len(t)):
            draw = mu[:, j] + s[:, j] * rng.standard_t(nu, M)
            o = np.argsort(draw); cw = np.cumsum(w[o])
            q = lambda a: draw[o][np.searchsorted(cw, a)]
            med, lo, hi = q(0.5), q(0.05), q(0.95)
            errs.append(y[j] - med); covs.append(lo - 1e-9 <= y[j] <= hi + 1e-9)
    err, cov = np.array(errs), np.array(covs)
    np.savez_compressed(OUT + f"pred_{PFX}{FOLD}.npz", err=err, cov=cov)
    res.update(n_pat=int(te.pid.nunique()), n_pred=int(len(err)), med_abs_err=float(np.median(np.abs(err))), mean_abs_err=float(np.abs(err).mean()),
               cov90=float(cov.mean()), **{"share_le_0.1": float((np.abs(err) <= 0.1 + 1e-9).mean())})
json.dump(res, open(OUT + f"fold_{PFX}{FOLD}.json", "w"), indent=1)
print(json.dumps(res, indent=1))
