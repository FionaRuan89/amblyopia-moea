"""Step 6 (RQ4): adaptive follow-up schedules.
Genes (7 integers): d1 in [2,16] (first interval); A, B, C in [2,16] = next interval after a visit whose VA improved by
>=2 lines / exactly 1 line / <=0 lines versus the previous visit; L in {1,2}; m in {1,2,3}; Tmin in [0,16].
Static fixed-interval schedules are the special case d1=A=B=C.  Same simulator, objectives, selection rule and held-out
evaluation as optimize_static.py.  Comparisons: adaptive-selected vs comparator (RQ2 metrics) and vs the static-selected
schedule (paired over the same posterior replicates); hypervolume of adaptive vs static NSGA-II runs.
usage: python optimize_adaptive.py <root> <n_jobs> <n_eval> <n_seeds> [n_train] [n_rep]
"""
import os, sys, time, pickle
ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]] + ARGS
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optimize_static as E
OUT = E.ROOT + "results/adaptive/"; os.makedirs(OUT, exist_ok=True)
XL = np.array([2, 2, 2, 2, 1, 1, 0]); XU = np.array([16, 16, 16, 16, 2, 3, 16])
H, LINE, KMAX = E.H, E.LINE, E.KMAX

def evaluate_adaptive(X, C, chunk=50, H_=None, return_vis=False):
    H_ = H if H_ is None else H_
    X = np.rint(np.atleast_2d(X)).astype(np.float32); out = []
    n = len(C["G"])
    for i in range(0, len(X), chunk):
        Y = X[i:i + chunk]; p = len(Y)
        d1, A, B, Cc, Lv, mv, Tm = (Y[:, k][:, None] for k in range(7))
        prev = np.round(C["V0"] + C["noise"][:, 0], 1)[None, :].repeat(p, 0)
        best = prev.copy()
        t = np.repeat(d1, n, 1)
        run = np.zeros((p, n), np.int8); stop = np.full((p, n), H_, np.float32); nvis = np.zeros((p, n), np.float32)
        done = np.zeros((p, n), bool); rule = np.zeros((p, n), bool)
        for j in range(KMAX - 1):
            act = ~done & (t <= H_ + 1e-6)
            if not act.any(): break
            y = np.round(C["V0"][None, :] - C["G"][None, :] * (1 - np.exp(-t / C["tau"][None, :])) + C["noise"][None, :, j + 1], 1)
            nvis += act
            imp = y <= best - LINE * Lv + 1e-6
            count = act & ~imp & (t >= Tm)
            run = np.where(act & imp, 0, np.where(count, run + 1, run)).astype(np.int8)
            best = np.where(act, np.minimum(best, y), best)
            newly = act & (run >= mv); stop = np.where(newly, t, stop); rule |= newly; done |= newly | ~act
            gain = np.rint((prev - y) / LINE)
            nxt = np.where(gain >= 2, A, np.where(gain >= 1, B, Cc))
            prev = np.where(act, y, prev); t = np.where(act, t + nxt, t)
        f1 = nvis.mean(1); f2 = np.maximum(0, stop - C["tstar"][None, :]).mean(1); f3 = (rule & (stop < C["tstar"][None, :])).mean(1)
        out.append(np.column_stack([f1, f2, f3]))
    return np.vstack(out)

def optimise(seed, C):
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.algorithms.moo.nsga2 import NSGA2
    arc = {"X": np.zeros((0, 7)), "F": np.zeros((0, 3))}
    class Pr(Problem):
        def __init__(s): super().__init__(n_var=7, n_obj=3, xl=XL, xu=XU, vtype=int)
        def _evaluate(s, X, out, *a, **k):
            X = np.rint(X); F = evaluate_adaptive(X, C); out["F"] = F
            arc["X"], arc["F"] = E.nondom_big(np.vstack([arc["X"], X]), np.vstack([arc["F"], F]))
    ops = dict(sampling=IntegerRandomSampling(), crossover=SBX(prob=0.9, eta=3, vtype=float, repair=RoundingRepair()),
               mutation=PM(eta=3, vtype=float, repair=RoundingRepair()), eliminate_duplicates=True)
    minimize(Pr(), NSGA2(pop_size=200, **ops), ("n_eval", E.NEVAL), seed=seed, verbose=False)
    return arc["X"], arc["F"]

def task(a):
    fold, seed = a; ck = OUT + f"runs/{fold}_{seed}.pkl"
    if os.path.exists(ck): return pickle.load(open(ck, "rb"))
    t0 = time.time(); C = E.train_cohort("all", fold); X, F = optimise(seed, C)
    r = dict(fold=fold, seed=seed, X=X, F=F, hv_train=E.hv(F), sec=time.time() - t0)
    os.makedirs(OUT + "runs", exist_ok=True); pickle.dump(r, open(ck + ".tmp", "wb")); os.replace(ck + ".tmp", ck)
    return r

def describe(x):
    x = np.rint(x).astype(int)
    return (f"first visit {x[0]} wk; next interval {x[1]} wk if improved >=2 lines, {x[2]} wk if 1 line, {x[3]} wk if no gain; "
            f"improvement >= {x[4]} line(s); stop after {x[5]} non-improving visit(s); Tmin {x[6]} wk")

def test_eval(fold, adaptive, static):
    """same replicates/seed as optimize_static.test_eval (rng 99): columns primary f1-f3, alt-definition f1-f3"""
    D = E.draws("ALL"); ti = list(D["trials"]).index(fold); rng = np.random.default_rng(99); R = []
    for r in range(E.NREP):
        C = E.cohort(D, 2000, rng, E.test_cov(fold), trial_idx=ti, one_draw=True); Ca = dict(C, tstar=C["tstar_alt"])
        a = np.concatenate([evaluate_adaptive(adaptive, C), evaluate_adaptive(adaptive, Ca)], 1)
        s = np.concatenate([E.evaluate(np.vstack([static, E.CMP]), C), E.evaluate(np.vstack([static, E.CMP]), Ca)], 1)
        R.append(np.vstack([a, s]))
    R = np.array(R); return {"adaptive": R[:, 0], "static": R[:, 1], "comparator": R[:, 2]}

def hv_test_task(a):
    key, X, fold, kind = a
    D = E.draws("ALL"); ti = list(D["trials"]).index(fold)
    C = E.cohort(D, 5000, np.random.default_rng(7), E.test_cov(fold), trial_idx=ti)
    F = evaluate_adaptive(X, C) if kind == "adaptive" else E.evaluate(X, C)
    k = E.nondom(F); return key, E.hv(F[k])

if __name__ == "__main__":
    from multiprocessing import Pool
    from scipy.stats import mannwhitneyu
    t0 = time.time()
    # sanity: adaptive evaluator with A=B=C must equal the static evaluator
    Ct = E.train_cohort("all", "ATS5"); Ct = {k: v[:3000] for k, v in Ct.items()}
    for d in (3, 5, 9, 14):
        for m in (1, 2, 3):
            fa = evaluate_adaptive(np.array([d, d, d, d, 1, m, 4.]), Ct)[0]; fs = E.evaluate(np.array([d] * 6 + [1, m, 4.]), Ct)[0]
            assert np.allclose(fa, fs, atol=1e-5), (d, m, fa, fs)
    print("sanity check passed: adaptive(d1=A=B=C) == static", flush=True)
    folds = E.MAIN + ["ALL"]
    if os.path.exists(OUT + "runs.pkl"): runs = pickle.load(open(OUT + "runs.pkl", "rb"))
    else:
        with Pool(E.NJ) as pool: runs = pool.map(task, [(f, s) for f in folds for s in range(E.NSEED)], chunksize=1)
        pickle.dump(runs, open(OUT + "runs.pkl", "wb"))
    if os.environ.get("RUNS_ONLY") == "1": sys.exit(0)
    print(f"{len(runs)} adaptive runs in {(time.time()-t0)/60:.1f} min", flush=True)
    prim = pickle.load(open(E.OUT + "results.pkl", "rb")); runs_static = pickle.load(open(E.OUT + "runs.pkl", "rb"))
    L = ["RQ4: adaptive schedules, NSGA-II x %d seeds, %d evaluations, plateau definition %s" % (E.NSEED, E.NEVAL, E.TSTAR), ""]
    sel = {}
    for f in folds:
        R = [r for r in runs if r["fold"] == f]
        X, F = E.nondom_big(np.vstack([r["X"] for r in R]), np.vstack([r["F"] for r in R]))
        C = E.train_cohort("all", f); X = np.unique(np.rint(X), axis=0); X, F = E.nondom_big(X, evaluate_adaptive(X, C))
        c = E.evaluate(E.CMP, C)[0]; j = E.select(X, F, c)
        if j is None: j = int(np.argmin(F[:, 1]))
        sel[f] = dict(x=X[j], train=F[j], cmp_train=c, front=len(F))
        np.savez_compressed(OUT + f"front_{f}.npz", X=X, F=F)
        L.append(f"[{f}] merged adaptive front {len(F)} | comparator train {np.round(c,3)} | selected train {np.round(F[j],3)} | {describe(X[j])}")
    L += ["", "Held-out (same 200 posterior replicates x 2000 patients as the static analysis)"]
    res = {}
    for f in E.MAIN:
        res[f] = test_eval(f, sel[f]["x"], prim["sel"][("all", f)]["x"])
        L.append(f"[{f}] " + " | ".join(f"{k} {np.round(v.mean(0)[:3],2).tolist()}" for k, v in res[f].items()))
    w = np.array([len(E.test_cov(f)) for f in E.MAIN], float); w /= w.sum()
    pooled = {k: sum(w[i] * res[f][k] for i, f in enumerate(E.MAIN)) for k in res[E.MAIN[0]]}
    for ref in ("comparator", "static"):
        for cols, lab in ((slice(0, 3), "primary plateau definition"), (slice(3, 6), "sensitivity plateau definition")):
            d = (pooled["adaptive"] - pooled[ref])[:, cols]; lo, hi = np.percentile(d, 2.5, 0), np.percentile(d, 97.5, 0)
            L.append(f"pooled adaptive - {ref:10s} [{lab}]: f1 {d.mean(0)[0]:+.2f} [{lo[0]:+.2f}, {hi[0]:+.2f}]  "
                     f"f2 {d.mean(0)[1]:+.2f} [{lo[1]:+.2f}, {hi[1]:+.2f}] wk  f3 {d.mean(0)[2]:+.3f} [{lo[2]:+.3f}, {hi[2]:+.3f}]"
                     + (f"  RQ2-type criterion: {'MET' if (hi[0] < 0 and hi[1] < 2 and hi[2] < 0.03) else 'NOT MET'}" if ref == "comparator" else ""))
    for k in ("adaptive", "static", "comparator"):
        L.append(f"pooled {k:10s}: f1 {pooled[k].mean(0)[0]:.2f}  f2 {pooled[k].mean(0)[1]:.2f}  f3 {pooled[k].mean(0)[2]:.3f}")
    a = sel["ALL"]; L += ["", f"Adaptive schedule, full-data posterior: {describe(a['x'])}", f"  train {np.round(a['train'],3)} vs comparator {np.round(a['cmp_train'],3)}"]
    # hypervolume: adaptive vs static NSGA-II
    s3 = [r for r in runs_static if r["stratum"] == "all" and r["alg"] == "NSGA2" and r["fold"] in E.MAIN]
    jobs = [((r["fold"], "adaptive", r["seed"]), r["X"], r["fold"], "adaptive") for r in runs if r["fold"] in E.MAIN]
    jobs += [((r["fold"], "static", r["seed"]), r["X"], r["fold"], "static") for r in s3]
    with Pool(E.NJ) as pool: hvt = dict(pool.map(hv_test_task, jobs, chunksize=1))
    rows = [dict(fold=r["fold"], kind="adaptive", seed=r["seed"], hv_train=r["hv_train"], hv_test=hvt[(r["fold"], "adaptive", r["seed"])]) for r in runs if r["fold"] in E.MAIN]
    rows += [dict(fold=r["fold"], kind="static", seed=r["seed"], hv_train=r["hv_train"], hv_test=hvt[(r["fold"], "static", r["seed"])]) for r in s3]
    Hd = pd.DataFrame(rows); Hd.to_csv(OUT + "hv_adaptive_vs_static.csv", index=False)
    L += ["", "Hypervolume, median over seeds (adaptive vs static NSGA-II)", Hd.groupby(["fold", "kind"])[["hv_train", "hv_test"]].median().unstack("kind").round(4).to_string()]
    for mtr in ("hv_train", "hv_test"):
        for f in E.MAIN:
            a_ = Hd[(Hd.fold == f) & (Hd.kind == "adaptive")][mtr]; s_ = Hd[(Hd.fold == f) & (Hd.kind == "static")][mtr]
            L.append(f"  {mtr} {f}: adaptive - static median {a_.median() - s_.median():+.4f}, Mann-Whitney p = {mannwhitneyu(a_, s_).pvalue:.2g}")
    print("\n".join(L)); open(OUT + "summary.txt", "w").write("\n".join(L))
    pickle.dump(dict(sel=sel, res=res), open(OUT + "results.pkl", "wb"))
