"""Step 5 (RQ3): six further MOEAs, run under exactly the protocol of optimize_static.py (same 9-gene integer encoding, SBX(0.9, eta 3) + PM(eta 3) + rounding repair,
60,000 evaluations, external archive, 30 seeds x 4 LOTO folds, same training cohorts and held-out HV evaluation).
Algorithms run here: SPEA2, RVEA, AGE-MOEA, AGE-MOEA-II  (pymoo implementations)
                     IBEA (I_eps+; Zitzler & Kuenzli 2004) and SPEA2+SDE (Li et al. 2014)  (implemented here on pymoo's GA framework)
SPEA2 uses pymoo's SPEA2 with its survival corrected: pymoo 0.6.1.5 computes the strength S(i) with sum(axis=0), i.e. the
number of solutions dominating i instead of the number i dominates (Zitzler et al. 2001); fixed here and in SPEA2+SDE.
Combined with the runs of optimize_static.py (NSGA-II, NSGA-III, SMS-EMOA, random search) -> 9 MOEAs + random search.
usage: python compare_algorithms.py <root> <n_jobs> <n_eval> <n_seeds> [n_train] [n_rep]   (env RUNS_ONLY=1 to skip the summary)
"""
import os, sys, time, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optimize_static as E
OUT = E.ROOT + "results/algorithms/"; os.makedirs(OUT + "runs", exist_ok=True)
NEW = ["SPEA2", "RVEA", "AGEMOEA", "AGEMOEA2", "IBEA", "SPEA2SDE"]
ALL10 = ["NSGA2", "NSGA3", "SMSEMOA"] + NEW
LABEL = {"NSGA2": "NSGA-II", "NSGA3": "NSGA-III", "SMSEMOA": "SMS-EMOA", "SPEA2": "SPEA2", "RVEA": "RVEA",
         "AGEMOEA": "AGE-MOEA", "AGEMOEA2": "AGE-MOEA-II", "IBEA": "IBEA", "SPEA2SDE": "SPEA2+SDE", "RANDOM": "random search"}

from pymoo.core.survival import Survival
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.util.dominator import Dominator

# ---------------- IBEA (Zitzler & Kuenzli 2004), additive epsilon indicator, kappa = 0.05 ----------------
class IBEASurvival(Survival):
    def __init__(self, kappa=0.05):
        super().__init__(filter_infeasible=True); self.kappa = kappa
    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        F = pop.get("F").astype(float); lo, hi = F.min(0), F.max(0); Fn = (F - lo) / np.where(hi > lo, hi - lo, 1)
        I = (Fn[:, None, :] - Fn[None, :, :]).max(2)          # I[i, j] = I_eps+(x_i, x_j)
        c = max(np.abs(I).max(), 1e-12); Ex = np.exp(-I / (c * self.kappa)); np.fill_diagonal(Ex, 0)
        fit = -Ex.sum(0)                                        # fitness of j = sum_i -exp(-I(i, j)/(c k))
        alive = np.ones(len(pop), bool)
        while alive.sum() > n_survive:
            w = np.where(alive, fit, np.inf).argmin(); alive[w] = False; fit += Ex[w]
        S = np.flatnonzero(alive); pop[S].set("fit", fit[S]); return pop[S]

def ibea_tournament(pop, P, **kwargs):
    f = np.array([0.0 if v is None else v for v in pop.get("fit")], float)
    return np.where(f[P[:, 0]] >= f[P[:, 1]], P[:, 0], P[:, 1])

# ---------------- SPEA2+SDE (Li, Yang & Liu 2014): SPEA2 with shift-based density estimation ----------------
class SPEA2SDESurvival(Survival):
    def __init__(self):
        super().__init__(filter_infeasible=True)
    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        F = pop.get("F").astype(float); lo, hi = F.min(0), F.max(0); Fn = (F - lo) / np.where(hi > lo, hi - lo, 1)
        M = Dominator().calc_domination_matrix(F); S = (M == 1).sum(1); R = ((M == -1) * S).sum(1)   # M[i,j] = 1: i dominates j
        # shifted distance: density of i measured to max(F_j, F_i)
        D2 = np.sqrt((np.maximum(Fn[None, :, :] - Fn[:, None, :], 0) ** 2).sum(2)); np.fill_diagonal(D2, np.inf)
        k = min(int(np.sqrt(len(pop))), len(pop) - 1); dens = 1 / (np.sort(D2, 1)[:, k] + 2); fitv = R + dens
        pop.set(SPEA_F=fitv)
        surv = list(np.flatnonzero(np.all(M >= 0, axis=1)))
        if len(surv) < n_survive:
            rem = np.array([i for i in range(len(pop)) if i not in set(surv)]); surv += list(rem[fitv[rem].argsort()][:n_survive - len(surv)])
        while len(surv) > n_survive:
            sub = D2[np.ix_(surv, surv)]; sd = np.sort(sub, 1)                     # truncation: lexicographic nearest SDE distances
            i = int(np.lexsort((sd[:, 1], sd[:, 0]))[0]) if sd.shape[1] > 1 else 0
            surv.pop(i)
        return pop[surv]

from pymoo.algorithms.moo.spea2 import SPEA2Survival
class SPEA2SurvivalFixed(SPEA2Survival):
    """pymoo 0.6.1.5 SPEA2Survival._do with the strength S(i) = number of solutions that i dominates (sum over axis 1)."""
    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        from pymoo.util.misc import vectorized_cdist
        from pymoo.algorithms.moo.nsga3 import HyperplaneNormalization
        F = pop.get("F").astype(float, copy=False); M = Dominator().calc_domination_matrix(F)
        S = (M == 1).sum(axis=1); R = ((M == -1) * S).sum(axis=1)
        k = int(np.sqrt(len(pop))); k = len(pop) - 1 if k >= len(pop) else k
        if self.normalize:
            if self.norm is None: self.norm = HyperplaneNormalization(F.shape[1])
            self.norm.update(F); ideal, nadir = self.norm.ideal_point, self.norm.nadir_point
            dists = vectorized_cdist((F - ideal) / (nadir - ideal), (F - ideal) / (nadir - ideal), fill_diag_with_inf=True)
        else:
            dists = vectorized_cdist(F, F, fill_diag_with_inf=True)
        D = 1 / (np.sort(dists, axis=1)[:, k] + 2); SPEA_F = R + D
        pop.set(SPEA_F=SPEA_F, SPEA_R=R, SPEA_D=D)
        survivors = list(np.where(np.all(M >= 0, axis=1))[0])
        if self.normalize:
            I = vectorized_cdist(self.norm.extreme_points, F[survivors]).argmin(axis=1); pop[survivors][I].set("SPEA_F", -1.0)
        H_ = set(survivors); rem = np.array([i for i in range(len(pop)) if i not in H_])
        if len(survivors) < n_survive:
            survivors.extend(rem[SPEA_F[rem].argsort()][:n_survive - len(survivors)])
        elif len(survivors) > n_survive:
            while len(survivors) > n_survive:
                i = dists[survivors][:, survivors].min(axis=1).argmin(); survivors = [survivors[j] for j in range(len(survivors)) if j != i]
        return pop[survivors]

def spea_tournament(pop, P, **kwargs):
    f = pop.get("SPEA_F").astype(float); return np.where(f[P[:, 0]] <= f[P[:, 1]], P[:, 0], P[:, 1])

def make(alg, ops):
    from pymoo.util.ref_dirs import get_reference_directions
    rd = get_reference_directions("das-dennis", 3, n_partitions=19)   # 210 directions ~ population 200
    if alg == "RVEA":
        from pymoo.algorithms.moo.rvea import RVEA
        return RVEA(ref_dirs=rd, sampling=ops["sampling"], crossover=ops["crossover"], mutation=ops["mutation"], eliminate_duplicates=True)
    if alg == "SPEA2":
        from pymoo.algorithms.moo.spea2 import SPEA2; return SPEA2(pop_size=200, survival=SPEA2SurvivalFixed(normalize=True), **ops)
    if alg == "AGEMOEA":
        from pymoo.algorithms.moo.age import AGEMOEA; return AGEMOEA(pop_size=200, **ops)
    if alg == "AGEMOEA2":
        from pymoo.algorithms.moo.age2 import AGEMOEA2; return AGEMOEA2(pop_size=200, **ops)
    if alg == "IBEA":
        return GeneticAlgorithm(pop_size=200, selection=TournamentSelection(func_comp=ibea_tournament), survival=IBEASurvival(), advance_after_initial_infill=True, **ops)
    if alg == "SPEA2SDE":
        return GeneticAlgorithm(pop_size=200, selection=TournamentSelection(func_comp=spea_tournament), survival=SPEA2SDESurvival(), advance_after_initial_infill=True, **ops)
    raise ValueError(alg)

def optimise(alg, seed, C):
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.repair.rounding import RoundingRepair
    arc = {"X": np.zeros((0, 9)), "F": np.zeros((0, 3))}; buf = {"X": [], "F": []}
    class Pr(Problem):
        def __init__(s): super().__init__(n_var=9, n_obj=3, xl=E.XL, xu=E.XU, vtype=int)
        def _evaluate(s, X, out, *a, **k):
            X = np.rint(X); F = E.evaluate(X, C); out["F"] = F; buf["X"].append(X); buf["F"].append(F)
            if sum(len(x) for x in buf["X"]) >= 2000:          # batched archive update (identical result, cheaper)
                arc["X"], arc["F"] = E.nondom_big(np.vstack([arc["X"]] + buf["X"]), np.vstack([arc["F"]] + buf["F"])); buf["X"], buf["F"] = [], []
    ops = dict(sampling=IntegerRandomSampling(), crossover=SBX(prob=0.9, eta=3, vtype=float, repair=RoundingRepair()),
               mutation=PM(eta=3, vtype=float, repair=RoundingRepair()))
    A = make(alg, ops if alg == "RVEA" else dict(ops, eliminate_duplicates=True))
    minimize(Pr(), A, ("n_eval", E.NEVAL), seed=seed, verbose=False)
    if buf["X"]: arc["X"], arc["F"] = E.nondom_big(np.vstack([arc["X"]] + buf["X"]), np.vstack([arc["F"]] + buf["F"]))
    return arc["X"], arc["F"]

def task(a):
    fold, alg, seed = a; ck = OUT + f"runs/{fold}_{alg}_{seed}.pkl"
    if os.path.exists(ck): return pickle.load(open(ck, "rb"))
    t0 = time.time(); C = E.train_cohort("all", fold); X, F = optimise(alg, seed, C)
    r = dict(stratum="all", fold=fold, alg=alg, seed=seed, X=X, F=F, hv_train=E.hv(F), sec=time.time() - t0)
    pickle.dump(r, open(ck + ".tmp", "wb")); os.replace(ck + ".tmp", ck); return r

def hv_task(a):
    key, ck = a; r = pickle.load(open(ck, "rb")); return key, E.hv_test_task((key, r["X"], r["fold"]))[1]

if __name__ == "__main__":
    from multiprocessing import Pool
    from scipy.stats import wilcoxon, friedmanchisquare
    t0 = time.time()
    tasks = [(f, a, s) for a in NEW for f in E.MAIN for s in range(E.NSEED)]
    tasks.sort(key=lambda t: t[1] not in ("AGEMOEA", "AGEMOEA2", "IBEA", "SPEA2SDE"))   # heavier survivals first
    with Pool(E.NJ) as pool: pool.map(task, tasks, chunksize=1)
    print(f"{len(tasks)} new runs done in {(time.time()-t0)/60:.1f} min", flush=True)
    if os.environ.get("RUNS_ONLY") == "1": sys.exit(0)
    # held-out hypervolume of these runs
    keys = [((f, a, s), OUT + f"runs/{f}_{a}_{s}.pkl") for (f, a, s) in tasks]
    with Pool(E.NJ) as pool: hvt = dict(pool.map(hv_task, keys, chunksize=1))
    rows = []
    for (f, a, s), ck in keys:
        r = pickle.load(open(ck, "rb")); rows.append(dict(fold=f, alg=a, seed=s, hv_train=r["hv_train"], hv_test=hvt[(f, a, s)], sec=r["sec"], n_front=len(r["F"])))
    H = pd.concat([pd.read_csv(E.OUT + "hv_per_run.csv"), pd.DataFrame(rows)], ignore_index=True); H.to_csv(OUT + "hv_all_algorithms.csv", index=False)
    algs = ALL10 + ["RANDOM"]
    L = ["RQ3: 9 MOEAs + random search, 30 seeds x 4 LOTO folds, 60,000 evaluations each", ""]
    for m in ("hv_train", "hv_test"):
        piv = H.pivot_table(index=["fold", "seed"], columns="alg", values=m)[algs]
        rk = piv.rank(axis=1, ascending=False)
        L += [f"--- {m} ---", "median by fold:", H.pivot_table(index="fold", columns="alg", values=m, aggfunc="median")[algs].round(4).rename(columns=LABEL).to_string(),
              "mean rank over 120 (fold, seed) blocks (1 = best): " + ", ".join(f"{LABEL[a]} {rk[a].mean():.2f}" for a in rk.mean().sort_values().index)]
        st = friedmanchisquare(*[piv[a] for a in ALL10]); L.append(f"Friedman test across the 9 MOEAs: chi2 = {st.statistic:.1f}, p = {st.pvalue:.2g}")
        others = [a for a in algs if a != "NSGA2"]; ps = {o: wilcoxon(piv["NSGA2"], piv[o]).pvalue for o in others}
        order = sorted(ps, key=ps.get); adj, prev = {}, 0
        for i, o in enumerate(order): prev = max(prev, min(1.0, ps[o] * (len(order) - i))); adj[o] = prev
        for o in others: L.append(f"  NSGA-II - {LABEL[o]:13s} median diff {(piv['NSGA2'] - piv[o]).median():+.4f}  Holm p = {adj[o]:.2g}")
    L += ["", "median run time (min) per algorithm: " + ", ".join(f"{LABEL[a]} {H[H.alg == a].sec.median()/60:.1f}" for a in algs),
          "median archive size: " + ", ".join(f"{LABEL[a]} {H[H.alg == a].n_front.median():.0f}" for a in algs)]
    print("\n".join(L)); open(OUT + "summary.txt", "w").write("\n".join(L))
