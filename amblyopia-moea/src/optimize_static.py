"""Step 4 (RQ2, RQ3): multi-objective optimisation of static follow-up schedules on simulated cohorts.
Also used as a module (simulator, evaluator, selection rule, held-out evaluation) by the later steps.
Genes (9 integers): d1..d6 in [2,16] wk (d6 repeats to H=52), L in {1,2} lines, m in {1,2,3}, Tmin in [0,16] wk.
Objectives: f1 mean visits, f2 mean delay max(0, stop - t*), f3 premature-stop rate (rule stop before t*).
Comparator: the PEDIG trial rule (ATS5 spectacle phase): a visit every 5 wk, follow-up ends at the first visit at which the
amblyopic-eye VA has not improved by >=1 line over the prior visit, confirmed by a same-day retest (the better of the two
measurements is recorded).  It is coded as m = 0, outside the search space (m in {1,2,3}).
Selection: minimise f1 subject to f2, f3 <= comparator; among schedules within TOL_F1 visits of that minimum, take the lowest f3
(then f2).
Strata: "all" (primary; LOTO folds ATS5/13/15/16 + ALL for the recommended schedule),
        "spec" (folds ATS5, ATS13), "amb" (patching/atropine; folds ATS15, ATS16)  [secondary].
usage: python optimize_static.py <root> <n_jobs> <n_eval> <n_seeds> [n_train] [n_rep]
Environment: TSTAR=win12 (primary plateau definition, default) or TSTAR=inf.
"""
import os, sys, time, json, pickle
import numpy as np, pandas as pd
ROOT = sys.argv[1].rstrip("/") + "/"; NJ = int(sys.argv[2]); NEVAL = int(sys.argv[3]); NSEED = int(sys.argv[4])
NTR = int(sys.argv[5]) if len(sys.argv) > 5 else 20000; NREP = int(sys.argv[6]) if len(sys.argv) > 6 else 200
MODEL_DIR = ROOT + "results/trajectory_model/"; OUT = ROOT + "results/static/"; os.makedirs(OUT, exist_ok=True)
H, LINE, NU, KMAX, WIN = 52.0, 0.1, 4.0, 27, 12.0
RHO_RT = 0.25   # correlation of a same-day retest error with the first measurement; matches the SD (0.128 logMAR) of the
                # 1,852 same-day test-retest differences in derived/retest.csv (see check_simulator.py)
TOL_F1 = 0.1    # selection tolerance on f1 (visits)
TSTAR = os.environ.get("TSTAR", "win12")   # primary: expected gain over the next 12 wk < 1 line; "inf": remaining total gain < 1 line
MAIN = ["ATS5", "ATS13", "ATS15", "ATS16"]
TXS = ["patching", "atropine"]
XL = np.array([2] * 6 + [1, 1, 0]); XU = np.array([16] * 6 + [2, 3, 16])
CMP = np.array([5] * 6 + [1, 0, 0], float)   # PEDIG trial rule (m = 0 code, see above)
SECOND = {"every5_m2": [5] * 6 + [1, 2, 0], "every4_m2": [4] * 6 + [1, 2, 0], "every6_m2": [6] * 6 + [1, 2, 0], "every9_m2": [9] * 6 + [1, 2, 0]}
P = pd.read_csv(ROOT + "derived/pts.csv"); P = P[P.role.isin(["main", "aux"])].copy()
P["age_c"] = (P.age.fillna(P.age.median()) - 5.5) / 2.0
STRATA = {"all": ["spectacles", "patching", "atropine"], "spec": ["spectacles"], "amb": ["patching", "atropine"]}
FOLDS = {"all": MAIN + ["ALL"], "spec": ["ATS5", "ATS13"], "amb": ["ATS15", "ATS16"]}
_D = {}
def draws(f):
    if f not in _D: _D[f] = dict(np.load(MODEL_DIR + f"draws_{f}.npz", allow_pickle=True))
    return _D[f]

# ---------------- cohorts ----------------
def cohort(D, n, rng, cov, trial_idx=None, one_draw=False):
    """cov: DataFrame with tx, age_c to resample.  trial_idx None -> new trial effect per patient."""
    S = len(D["muG"]); s = np.full(n, rng.integers(S)) if one_draw else rng.integers(S, size=n)
    c = cov.iloc[rng.integers(len(cov), size=n)]
    tx = np.column_stack([(c.tx == t).values for t in TXS]).astype(float); age = c.age_c.values
    if trial_idx is None:
        uV, uG, uT = (rng.standard_normal(n) * D[k][s] for k in ("tV", "tG", "tT"))
    else:
        uV, uG, uT = (D[z][s, trial_idx] * D[k][s] for z, k in (("uV_z", "tV"), ("uG_z", "tG"), ("uT_z", "tT")))
    V0 = D["muV"][s] + uV + D["sV"][s] * rng.standard_normal(n); vc = V0 - 0.6
    rho = D["rho"][s]; z1, z2 = rng.standard_normal(n), rng.standard_normal(n); zt = rho * z1 + np.sqrt(1 - rho ** 2) * z2
    G = D["muG"][s] + (tx * D["bG"][s]).sum(1) + D["bGV"][s] * vc + D["bGA"][s] * age + uG + D["sG"][s] * z1
    lt = D["muT"][s] + (tx * D["bT"][s]).sum(1) + D["bTV"][s] * vc + D["bTA"][s] * age + uT + D["sT"][s] * zt
    tau = np.exp(np.clip(lt, -3, 8))
    Gw = G * (1 - np.exp(-WIN / tau))
    t_win = np.where(Gw > LINE, tau * np.log(np.maximum(Gw, LINE) / LINE), 0.0)
    t_inf = np.where(G > LINE, tau * np.log(np.maximum(G, LINE) / LINE), 0.0)
    tstar = t_win if TSTAR == "win12" else t_inf
    sig = D["sig"][s, 0]
    noise = (rng.standard_t(NU, size=(n, KMAX)) * sig[:, None]).astype(np.float32)
    # same-day retest errors from a spawned stream, so that every other random number of the cohort is unchanged
    z = rng.spawn(1)[0].standard_t(NU, size=(n, KMAX)) * sig[:, None]
    noise_rt = (RHO_RT * noise + np.sqrt(1 - RHO_RT ** 2) * z).astype(np.float32)
    return dict(V0=V0.astype(np.float32), G=G.astype(np.float32), tau=tau.astype(np.float32), tstar=tstar.astype(np.float32),
                tstar_alt=(t_inf if TSTAR == "win12" else t_win).astype(np.float32), noise=noise, noise_rt=noise_rt)

def times_of(X):
    X = np.rint(np.atleast_2d(X)).astype(float); d = np.concatenate([X[:, :6], np.repeat(X[:, 5:6], KMAX - 7, 1)], 1)
    t = np.cumsum(d, 1); t[t > H + 1e-9] = np.inf; return t, X[:, 6], X[:, 7], X[:, 8]

def evaluate(X, C, chunk=100):
    out = []
    X = np.atleast_2d(X)
    for i in range(0, len(X), chunk):
        T, Lv, mv, Tm = times_of(X[i:i + chunk]); p = len(T); n = len(C["G"])
        pr = mv == 0; mv = np.where(pr, 1, mv)          # PEDIG trial rule rows
        best = np.round(C["V0"] + C["noise"][:, 0], 1)[None, :].repeat(p, 0); prev = best.copy()
        run = np.zeros((p, n), np.int8); stop = np.full((p, n), H, np.float32); nvis = np.zeros((p, n), np.float32)
        done = np.zeros((p, n), bool); rule = np.zeros((p, n), bool)
        for j in range(KMAX - 1):
            t = T[:, j]
            if np.all(np.isinf(t)): break
            tt = np.where(np.isinf(t), H, t).astype(np.float32)[:, None]
            y = np.round(C["V0"][None, :] - C["G"][None, :] * (1 - np.exp(-tt / C["tau"][None, :])) + C["noise"][None, :, j + 1], 1)
            act = ~done & np.isfinite(t)[:, None]
            nvis += act
            imp = y <= best - LINE * Lv[:, None] + 1e-6
            if pr.any():   # compare with the prior visit; if not improved, retest the same day and keep the better value
                y2 = np.round(C["V0"][None, :] - C["G"][None, :] * (1 - np.exp(-tt / C["tau"][None, :])) + C["noise_rt"][None, :, j + 1], 1)
                imp1 = y <= prev - LINE * Lv[:, None] + 1e-6
                yrec = np.where(imp1, y, np.minimum(y, y2))
                imp = np.where(pr[:, None], yrec <= prev - LINE * Lv[:, None] + 1e-6, imp)
                prev = np.where(act, yrec, prev)
            count = act & ~imp & (tt >= Tm[:, None])
            run = np.where(act & imp, 0, np.where(count, run + 1, run)).astype(np.int8)
            best = np.where(act, np.minimum(best, y), best)
            newly = act & (run >= mv[:, None]); stop = np.where(newly, tt, stop); rule |= newly; done |= newly
        f1 = nvis.mean(1); f2 = np.maximum(0, stop - C["tstar"][None, :]).mean(1); f3 = (rule & (stop < C["tstar"][None, :])).mean(1)
        out.append(np.column_stack([f1, f2, f3]))
    return np.vstack(out)

def nondom(F):
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    return NonDominatedSorting().do(F, only_non_dominated_front=True)
def nondom_big(X, F):
    F, i = np.unique(np.round(F, 6), axis=0, return_index=True); X = X[i]; cx, cf = X[:0], F[:0]
    for s in range(0, len(F), 4000):
        X2, F2 = np.vstack([cx, X[s:s + 4000]]), np.vstack([cf, F[s:s + 4000]]); k = nondom(F2); cx, cf = X2[k], F2[k]
    return cx, cf
def hv(F):
    from pymoo.indicators.hv import HV
    if not len(F): return 0.0
    G = np.column_stack([F[:, 0] / 8, F[:, 1] / 52, F[:, 2]]); G = G[(G < 1.1).all(1)]
    return float(HV(ref_point=np.array([1.1, 1.1, 1.1]))(G)) if len(G) else 0.0

def train_cov(stratum, fold):
    c = P[P.tx.isin(STRATA[stratum]) & (P.trial != fold)]
    return c[c.role == "main"] if stratum == "all" else c
def train_cohort(stratum, fold):
    return cohort(draws(fold), NTR, np.random.default_rng(11), train_cov(stratum, fold))

# ---------------- optimisation ----------------
def optimise(alg, seed, C):
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.util.ref_dirs import get_reference_directions
    arc = {"X": np.zeros((0, 9)), "F": np.zeros((0, 3))}
    if alg == "RANDOM":
        r = np.random.default_rng(seed)
        for _ in range(int(np.ceil(NEVAL / 2000))):
            X = r.integers(XL, XU + 1, size=(2000, 9)).astype(float); F = evaluate(X, C)
            arc["X"], arc["F"] = nondom_big(np.vstack([arc["X"], X]), np.vstack([arc["F"], F]))
        return arc["X"], arc["F"]
    class Pr(Problem):
        def __init__(s): super().__init__(n_var=9, n_obj=3, xl=XL, xu=XU, vtype=int)
        def _evaluate(s, X, out, *a, **k):
            X = np.rint(X); F = evaluate(X, C); out["F"] = F
            arc["X"], arc["F"] = nondom_big(np.vstack([arc["X"], X]), np.vstack([arc["F"], F]))
    ops = dict(sampling=IntegerRandomSampling(), crossover=SBX(prob=0.9, eta=3, vtype=float, repair=RoundingRepair()),
               mutation=PM(eta=3, vtype=float, repair=RoundingRepair()), eliminate_duplicates=True)
    if alg == "NSGA2":
        from pymoo.algorithms.moo.nsga2 import NSGA2; A = NSGA2(pop_size=200, **ops)
    elif alg == "NSGA3":
        from pymoo.algorithms.moo.nsga3 import NSGA3; A = NSGA3(pop_size=200, ref_dirs=get_reference_directions("das-dennis", 3, n_partitions=12), **ops)
    else:
        from pymoo.algorithms.moo.sms import SMSEMOA; A = SMSEMOA(pop_size=200, **ops)
    minimize(Pr(), A, ("n_eval", NEVAL), seed=seed, verbose=False)
    return arc["X"], arc["F"]

def task(a):
    stratum, fold, alg, seed = a; t0 = time.time()
    ck = OUT + f"runs/{stratum}_{fold}_{alg}_{seed}.pkl"      # per-run checkpoint
    if os.path.exists(ck): return pickle.load(open(ck, "rb"))
    C = train_cohort(stratum, fold); X, F = optimise(alg, seed, C)
    r = dict(stratum=stratum, fold=fold, alg=alg, seed=seed, X=X, F=F, hv_train=hv(F), sec=time.time() - t0)
    os.makedirs(OUT + "runs", exist_ok=True); pickle.dump(r, open(ck + ".tmp", "wb")); os.replace(ck + ".tmp", ck)
    return r

def grid_front(C):
    Xs = []
    for d in range(2, 17):
        for L in (1, 2):
            for m in (1, 2, 3): Xs.append([d] * 6 + [L, m, 0])
    for d1 in range(2, 13):
        for k in range(1, 6):
            for d2 in range(2, 17):
                if d2 == d1: continue
                for m in (1, 2, 3): Xs.append([d1] * k + [d2] * (6 - k) + [1, m, 0])
    X = np.array(Xs, float); F = evaluate(X, C); k = nondom(F); return X[k], F[k], len(X)

def select(X, F, c, tol=TOL_F1):
    """epsilon-constraint: f2, f3 <= comparator; minimise f1, then lowest f3 (then f2) within tol visits of the minimum."""
    ok = np.flatnonzero((F[:, 1] <= c[1] + 1e-9) & (F[:, 2] <= c[2] + 1e-9))
    if not len(ok): return None
    band = ok[F[ok, 0] <= F[ok, 0].min() + tol + 1e-9]
    return band[np.lexsort((F[band, 1], F[band, 2]))[0]]

def describe(x):
    x = np.rint(x).astype(int)
    if x[7] == 0: return "PEDIG trial rule: every 5 wk; stop at the first visit without a >=1-line gain over the prior visit, confirmed by a same-day retest"
    return f"intervals {x[:6].tolist()} (then every {x[5]} wk); improvement >= {x[6]} line(s); stop after {x[7]} non-improving visit(s); Tmin {x[8]} wk"

def test_cov(fold): return P[P.trial == fold]
def test_eval(stratum, fold, scheds):
    """200 replicates x 2000 patients from the full-data posterior with trial `fold`'s own effects."""
    D = draws("ALL"); ti = list(D["trials"]).index(fold); rng = np.random.default_rng(99)
    names = list(scheds); Xs = np.array([scheds[k] for k in names], float); R = []
    for r in range(NREP):
        C = cohort(D, 2000, rng, test_cov(fold), trial_idx=ti, one_draw=True); Ca = dict(C, tstar=C["tstar_alt"])
        R.append(np.concatenate([evaluate(Xs, C), evaluate(Xs, Ca)], 1))
    R = np.array(R)   # (rep, sched, 6): primary f1-f3, alt-definition f1-f3
    return {k: R[:, i, :] for i, k in enumerate(names)}

def hv_test_task(a):
    key, X, fold = a
    D = draws("ALL"); ti = list(D["trials"]).index(fold)
    C = cohort(D, 5000, np.random.default_rng(7), test_cov(fold), trial_idx=ti)
    F = evaluate(X, C); k = nondom(F); return key, hv(F[k])   # whole archive

# ======================================================================= main
if __name__ == "__main__":
    from multiprocessing import Pool
    tasks = [("all", f, a, s) for f in MAIN for a in ("NSGA2", "NSGA3", "SMSEMOA", "RANDOM") for s in range(NSEED)]
    tasks += [("all", "ALL", "NSGA2", s) for s in range(NSEED)]
    tasks += [(st, f, "NSGA2", s) for st in ("spec", "amb") for f in FOLDS[st] for s in range(NSEED)]
    t0 = time.time()
    if os.path.exists(OUT + "runs.pkl"): runs = pickle.load(open(OUT + "runs.pkl", "rb"))
    else:
        order = sorted(range(len(tasks)), key=lambda i: tasks[i][2] != "SMSEMOA")   # longest jobs first
        with Pool(NJ) as pool: out = pool.map(task, [tasks[i] for i in order], chunksize=1)
        runs = [None] * len(tasks)
        for i, r in zip(order, out): runs[i] = r
        pickle.dump(runs, open(OUT + "runs.pkl", "wb"))
    print(f"{len(runs)} optimisation runs in {(time.time()-t0)/60:.1f} min", flush=True)

    L = []; sel = {}
    for st in ("all", "spec", "amb"):
        for f in FOLDS[st]:
            R = [r for r in runs if r["stratum"] == st and r["fold"] == f and r["alg"] == "NSGA2"]
            X, F = nondom_big(np.vstack([r["X"] for r in R]), np.vstack([r["F"] for r in R]))
            C = train_cohort(st, f); X = np.unique(np.rint(X), axis=0); X, F = nondom_big(X, evaluate(X, C))
            c = evaluate(CMP, C)[0]; j = select(X, F, c)
            if j is None: j = int(np.argmin(F[:, 1]))
            sel[(st, f)] = dict(x=X[j], train=F[j], cmp_train=c, front=len(F))
            L.append(f"[{st}/{f}] merged front {len(F)} | comparator train {np.round(c,3)} | selected train {np.round(F[j],3)} | {describe(X[j])}")
            np.savez_compressed(OUT + f"front_{st}_{f}.npz", X=X, F=F)
    print("\n".join(L), flush=True)

    # ---- RQ2 (primary, stratum all) + strata: paired held-out evaluation ----
    L.append("\n" + "=" * 100 + "\nRQ2: held-out trials (200 posterior replicates x 2000 simulated patients each)\n" + "=" * 100)
    res = {}
    for st in ("all", "spec", "amb"):
        for f in FOLDS[st]:
            if f == "ALL": continue
            sc = {"selected": sel[(st, f)]["x"], "comparator": CMP, **{k: np.array(v, float) for k, v in SECOND.items()}}
            res[(st, f)] = test_eval(st, f, sc)
            m = {k: v.mean(0) for k, v in res[(st, f)].items()}
            L.append(f"[{st}/{f}] " + " | ".join(f"{k} {np.round(v[:3],2).tolist()}" for k, v in m.items()))
    for st in ("all", "spec", "amb"):
        fs = [f for f in FOLDS[st] if f != "ALL"]; w = np.array([len(test_cov(f)) for f in fs], float); w /= w.sum()
        pooled = {k: sum(w[i] * res[(st, f)][k] for i, f in enumerate(fs)) for k in res[(st, fs[0])]}
        da = (pooled["selected"] - pooled["comparator"])[:, 3:]; la, ha = np.percentile(da, 2.5, 0), np.percentile(da, 97.5, 0)
        alt = f"  [sensitivity, plateau = remaining total gain < 1 line] diff f1 {da.mean(0)[0]:+.2f} f2 {da.mean(0)[1]:+.2f} [{la[1]:+.2f}, {ha[1]:+.2f}] f3 {da.mean(0)[2]:+.3f} [{la[2]:+.3f}, {ha[2]:+.3f}]"
        for k in pooled: pooled[k] = pooled[k][:, :3]
        d = pooled["selected"] - pooled["comparator"]; lo, hi = np.percentile(d, 2.5, 0), np.percentile(d, 97.5, 0)
        met = hi[0] < 0 and hi[1] < 2 and hi[2] < 0.03
        L += [f"\n[{st}] pooled held-out (weights {dict(zip(fs, np.round(w,2)))}):",
              f"  comparator f1 {pooled['comparator'].mean(0)[0]:.2f} visits, f2 {pooled['comparator'].mean(0)[1]:.2f} wk, f3 {pooled['comparator'].mean(0)[2]:.3f}",
              f"  selected   f1 {pooled['selected'].mean(0)[0]:.2f} visits, f2 {pooled['selected'].mean(0)[1]:.2f} wk, f3 {pooled['selected'].mean(0)[2]:.3f}",
              f"  difference f1 {d.mean(0)[0]:+.2f} [{lo[0]:+.2f}, {hi[0]:+.2f}]  f2 {d.mean(0)[1]:+.2f} [{lo[1]:+.2f}, {hi[1]:+.2f}] wk  f3 {d.mean(0)[2]:+.3f} [{lo[2]:+.3f}, {hi[2]:+.3f}]",
              f"  criterion (f1 upper < 0, f2 upper < +2 wk, f3 upper < +0.03): {'MET' if met else 'NOT MET'}" + ("   <- PRIMARY RQ2" if st == "all" else "   (secondary)"), alt]
    a = sel[("all", "ALL")]
    L += ["", f"Recommended schedule (full-data posterior, stratum all): {describe(a['x'])}",
          f"  train f1/f2/f3 {np.round(a['train'],3)} vs comparator {np.round(a['cmp_train'],3)}"]
    L.append(f"plateau definition (primary): {TSTAR}")

    # ---- RQ3: hypervolume ----
    from scipy.stats import wilcoxon
    R3 = [r for r in runs if r["stratum"] == "all" and r["fold"] in MAIN]
    with Pool(NJ) as pool:
        hvt = dict(pool.map(hv_test_task, [((r["fold"], r["alg"], r["seed"]), r["X"], r["fold"]) for r in R3], chunksize=1))
    rows = [dict(fold=r["fold"], alg=r["alg"], seed=r["seed"], hv_train=r["hv_train"], hv_test=hvt[(r["fold"], r["alg"], r["seed"])], sec=r["sec"], n_front=len(r["F"])) for r in R3]
    H3 = pd.DataFrame(rows); H3.to_csv(OUT + "hv_per_run.csv", index=False)
    L += ["", "=" * 100, "RQ3: hypervolume (median over seeds)", "=" * 100,
          H3.groupby(["fold", "alg"])[["hv_train", "hv_test", "n_front", "sec"]].median().unstack("alg").round(4).to_string()]
    for f in MAIN:
        Xg, Fg, ng = grid_front(train_cohort("all", f)); L.append(f"  {f}: exhaustive fixed/two-phase grid ({ng} schedules) hv_train {hv(Fg):.4f}")
    for metric in ("hv_train", "hv_test"):
        piv = H3.pivot_table(index=["fold", "seed"], columns="alg", values=metric); ps = {o: wilcoxon(piv["NSGA2"], piv[o]).pvalue for o in ("NSGA3", "SMSEMOA", "RANDOM")}
        order = sorted(ps, key=ps.get); adj = {}
        for r_, o in enumerate(order): adj[o] = min(1.0, max(ps[o] * (3 - r_), adj.get(order[r_ - 1], 0) if r_ else 0))
        for o in ("NSGA3", "SMSEMOA", "RANDOM"):
            L.append(f"  {metric}: NSGA2 - {o:8s} median diff {(piv['NSGA2'] - piv[o]).median():+.4f}, Holm p = {adj[o]:.2g}")
    print("\n".join(L)); open(OUT + "summary.txt", "w").write("\n".join(L))
    pickle.dump(dict(sel=sel, res=res), open(OUT + "results.pkl", "wb"))
