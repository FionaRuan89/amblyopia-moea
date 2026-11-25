"""Step 7: sensitivity analyses for the primary RQ2 pipeline (stratum "all", static schedules).
For every condition the whole pipeline is repeated (train cohort -> NSGA-II x n_seeds -> merged archive -> epsilon-constraint
selection -> held-out evaluation against the comparator under the same condition).  In addition the schedule selected in
the primary analysis for each fold is evaluated under the condition.
Conditions: noise15 (observation noise x1.5, less standardized testing), plat005 / plat015 (plateau threshold 0.05 / 0.15 logMAR instead
of 0.1), H36 (horizon 36 wk), noaux (trajectory model fitted on the 4 main trials only; requires NOAUX=1 fits).
Part 2: model-free replay of the recommended (full-data) and comparator schedules on the observed trajectories
(ATS12 weekly; ATS5/13/15/16 nearest visit within +-2 wk) -- descriptive.
usage: python sensitivity_and_replay.py <root> <n_jobs> <n_eval> <n_seeds> [n_train] [n_rep]
"""
import os, sys, time, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optimize_static as E
OUT = E.ROOT + "results/sensitivity/"; os.makedirs(OUT, exist_ok=True)
CONDS = ["noise15", "plat005", "plat015", "H36", "noaux"]

def set_cond(cond): E.H = 36.0 if cond == "H36" else 52.0

def modify(C, cond, G=None):
    C = dict(C)
    if cond == "noise15": C["noise"] = C["noise"] * 1.5; C["noise_rt"] = C["noise_rt"] * 1.5
    if cond in ("plat005", "plat015"):
        th = 0.05 if cond == "plat005" else 0.15; Gg, tau = C["G"].astype(float), C["tau"].astype(float)
        Gw = Gg * (1 - np.exp(-E.WIN / tau))
        t_win = np.where(Gw > th, tau * np.log(np.maximum(Gw, th) / th), 0.0)
        t_inf = np.where(Gg > th, tau * np.log(np.maximum(Gg, th) / th), 0.0)
        C["tstar"], C["tstar_alt"] = (t_win, t_inf) if E.TSTAR == "win12" else (t_inf, t_win)
        C["tstar"] = C["tstar"].astype(np.float32); C["tstar_alt"] = C["tstar_alt"].astype(np.float32)
    return C

def dname(cond, f): return ("noaux_" + f) if cond == "noaux" else f

def train_cohort(cond, f):
    set_cond(cond)
    C = E.cohort(E.draws(dname(cond, f)), E.NTR, np.random.default_rng(11), E.train_cov("all", f))
    return modify(C, cond)

def task(a):
    cond, f, seed = a; ck = OUT + f"runs/{cond}_{f}_{seed}.pkl"
    if os.path.exists(ck): return pickle.load(open(ck, "rb"))
    t0 = time.time(); C = train_cohort(cond, f); X, F = E.optimise("NSGA2", seed, C)
    r = dict(cond=cond, fold=f, seed=seed, X=X, F=F, sec=time.time() - t0)
    os.makedirs(OUT + "runs", exist_ok=True); pickle.dump(r, open(ck + ".tmp", "wb")); os.replace(ck + ".tmp", ck)
    return r

def test_eval(cond, f, Xs):
    set_cond(cond); D = E.draws(dname(cond, "ALL")); ti = list(D["trials"]).index(f); rng = np.random.default_rng(99); R = []
    for r in range(E.NREP):
        C = modify(E.cohort(D, 2000, rng, E.test_cov(f), trial_idx=ti, one_draw=True), cond); R.append(E.evaluate(Xs, C))
    return np.array(R)   # rep x sched x 3

# ---------------- model-free replay ----------------
def emp_tstar(t, y, smooth):
    ys = pd.Series(y).rolling(3, center=True, min_periods=1).mean().values if smooth else y
    for i in range(len(t)):
        later = (t > t[i]) & (t <= t[i] + E.WIN)
        if not later.any() or ys[later].min() > ys[i] - E.LINE + 1e-6: return t[i]
    return t[-1]

def replay(x, t, y, tol, ts):
    tv, Lv, mv, Tm = E.times_of(x); tv = tv[0][np.isfinite(tv[0])]; Lv, mv, Tm = Lv[0], mv[0], Tm[0]
    pr = mv == 0; mv = 1 if pr else mv        # PEDIG trial rule: compare with the prior visit (no retest in the observed data)
    best, prev, run, nvis = y[0], y[0], 0, 0
    for s in tv:
        if s > t[-1] + tol: break
        k = np.argmin(np.abs(t - s))
        if abs(t[k] - s) > tol: continue
        nvis += 1; v = y[k]
        if v <= (prev if pr else best) - E.LINE * Lv + 1e-6: run = 0
        elif s >= Tm: run += 1
        best = min(best, v); prev = v
        if run >= mv: return nvis, s, True, max(0.0, s - ts), s < ts
    end = t[-1]
    return nvis, end, False, max(0.0, end - ts), False

if __name__ == "__main__":
    from multiprocessing import Pool
    t0 = time.time()
    prim = pickle.load(open(E.OUT + "results.pkl", "rb"))
    if os.path.exists(OUT + "runs.pkl"): runs = pickle.load(open(OUT + "runs.pkl", "rb"))
    else:
        with Pool(E.NJ) as pool: runs = pool.map(task, [(c, f, s) for c in CONDS for f in E.MAIN for s in range(E.NSEED)], chunksize=1)
        pickle.dump(runs, open(OUT + "runs.pkl", "wb"))
    if os.environ.get("RUNS_ONLY") == "1": sys.exit(0)
    print(f"{len(runs)} runs in {(time.time()-t0)/60:.1f} min", flush=True)
    L = [f"Sensitivity analyses (primary plateau definition {E.TSTAR}; NSGA-II x {E.NSEED} seeds x {E.NEVAL} evaluations; "
         f"held-out {E.NREP} replicates x 2000)", ""]
    w = np.array([len(E.test_cov(f)) for f in E.MAIN], float); w /= w.sum(); summ = {}
    for cond in CONDS:
        res = {}
        for f in E.MAIN:
            R = [r for r in runs if r["cond"] == cond and r["fold"] == f]
            X, F = E.nondom_big(np.vstack([r["X"] for r in R]), np.vstack([r["F"] for r in R]))
            C = train_cohort(cond, f); X = np.unique(np.rint(X), axis=0); X, F = E.nondom_big(X, E.evaluate(X, C))
            c = E.evaluate(E.CMP, C)[0]; j = E.select(X, F, c)
            if j is None: j = int(np.argmin(F[:, 1]))
            x_prim = prim["sel"][("all", f)]["x"]
            res[f] = test_eval(cond, f, np.vstack([X[j], x_prim, E.CMP]))
            L.append(f"[{cond}/{f}] selected {E.describe(X[j])} | held-out sel {np.round(res[f][:,0].mean(0),2).tolist()} "
                     f"primary-sel {np.round(res[f][:,1].mean(0),2).tolist()} comparator {np.round(res[f][:,2].mean(0),2).tolist()}")
        P_ = sum(w[i] * res[f] for i, f in enumerate(E.MAIN))
        for k, lab in ((0, "re-optimised under condition"), (1, "primary schedule            ")):
            d = P_[:, k] - P_[:, 2]; lo, hi = np.percentile(d, 2.5, 0), np.percentile(d, 97.5, 0)
            met = hi[0] < 0 and hi[1] < 2 and hi[2] < 0.03; summ[(cond, k)] = (d.mean(0), lo, hi, met)
            L.append(f"  [{cond}] pooled {lab} - comparator: f1 {d.mean(0)[0]:+.2f} [{lo[0]:+.2f}, {hi[0]:+.2f}]  f2 {d.mean(0)[1]:+.2f} "
                     f"[{lo[1]:+.2f}, {hi[1]:+.2f}] wk  f3 {d.mean(0)[2]:+.3f} [{lo[2]:+.3f}, {hi[2]:+.3f}]  criterion {'MET' if met else 'NOT MET'}")
        L.append(f"  [{cond}] pooled comparator f1 {P_[:,2].mean(0)[0]:.2f} f2 {P_[:,2].mean(0)[1]:.2f} f3 {P_[:,2].mean(0)[2]:.3f}; "
                 f"re-optimised f1 {P_[:,0].mean(0)[0]:.2f} f2 {P_[:,0].mean(0)[1]:.2f} f3 {P_[:,0].mean(0)[2]:.3f}")
        L.append("")
    set_cond("base")
    # ---- model-free replay ----
    T = pd.read_csv(E.ROOT + "derived/traj.csv")
    rec = prim["sel"][("all", "ALL")]["x"]; sch = {"recommended": rec, "comparator": E.CMP, **{k: np.array(v, float) for k, v in E.SECOND.items()}}
    L += ["Model-free replay on observed trajectories (descriptive; empirical plateau = first visit after which no observation within",
          "the next 12 wk is >=1 line better; ATS12 uses a 3-point running mean). Columns: visits, delay (wk), premature-stop rate, rule-stop rate"]
    rows = []
    for trial, tol, smooth in (("ATS12", 0.5, True), ("ATS5", 2.0, False), ("ATS13", 2.0, False), ("ATS15", 2.0, False), ("ATS16", 2.0, False)):
        G_ = [g.sort_values("wk") for _, g in T[T.trial == trial].groupby("pid")]
        G_ = [g for g in G_ if len(g) >= 3]
        for name, x in sch.items():
            o = np.array([replay(x, g.wk.values, g.va.values, tol, emp_tstar(g.wk.values, g.va.values, smooth)) for g in G_], float)
            rows.append(dict(trial=trial, n=len(G_), schedule=name, visits=o[:, 0].mean(), delay=o[:, 3].mean(), premature=o[:, 4].mean(), rule_stop=o[:, 2].mean()))
    Rp = pd.DataFrame(rows); Rp.to_csv(OUT + "replay.csv", index=False)
    L.append(Rp.round(3).to_string(index=False))
    L.append(f"recommended schedule: {E.describe(rec)}")
    print("\n".join(L)); open(OUT + "summary.txt", "w").write("\n".join(L))
    pickle.dump(summ, open(OUT + "results.pkl", "wb"))
