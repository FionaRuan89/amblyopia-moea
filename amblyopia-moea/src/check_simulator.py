"""Check of the vectorised evaluator in optimize_static.py against an independent child-by-child implementation
of the visit schedule and plateau-declaration rule, on the full-data training cohort.
Schedules: recommended (full-data selection), the PEDIG-trial-rule comparator, the 5-weekly two-visit rule, and a third schedule.
Also reports the same-day retest calibration (SD of retest-minus-test differences, simulated vs derived/retest.csv).
usage: python check_simulator.py <root> [n_children]"""
import os, sys
import numpy as np
ROOT = sys.argv[1].rstrip("/") + "/"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
sys.argv = [sys.argv[0], ROOT, "1", "1", "1", str(N), "1"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optimize_static as E

def one_child(x, V0, G, tau, tstar, noise, noise_rt, H=E.H):
    """Follow one child visit by visit; returns (visits, stop time, rule stop)."""
    d = [int(round(v)) for v in x[:6]]; L, m, Tmin = int(round(x[6])), int(round(x[7])), x[8]
    best = round(float(V0 + noise[0]), 1); prev = best; run, t, k, nvis = 0, 0.0, 0, 0
    while True:
        t += d[min(k, 5)]; k += 1
        if t > H + 1e-9 or k >= E.KMAX: return nvis, H, False
        nvis += 1
        y = round(float(V0 - G * (1 - np.exp(-t / tau)) + noise[k]), 1)
        if m == 0:   # PEDIG trial rule: vs prior visit, retest the same day if not improved, keep the better value
            if y > prev - 0.1 * L + 1e-6: y = min(y, round(float(V0 - G * (1 - np.exp(-t / tau)) + noise_rt[k]), 1))
            stop = y > prev - 0.1 * L + 1e-6; prev = y
            if stop: return nvis, t, True
            continue
        if y <= best - 0.1 * L + 1e-6: run = 0
        elif t >= Tmin: run += 1
        best = min(best, y)
        if run >= m: return nvis, t, True

def objectives(x, C):
    out = np.array([one_child(x, C["V0"][i], C["G"][i], C["tau"][i], C["tstar"][i], C["noise"][i], C["noise_rt"][i]) for i in range(len(C["G"]))], float)
    nvis, stop, rule = out[:, 0], out[:, 1], out[:, 2].astype(bool)
    return np.array([nvis.mean(), np.maximum(0, stop - C["tstar"]).mean(), (rule & (stop < C["tstar"])).mean()])

if __name__ == "__main__":
    C = E.train_cohort("all", "ALL")
    import pandas as pd
    R = pd.read_csv(ROOT + "derived/retest.csv"); mu = C["V0"].astype(float)
    sim = np.round(mu + C["noise_rt"][:, 1], 1) - np.round(mu + C["noise"][:, 1], 1)
    print(f"same-day retest difference SD: observed {np.std(R.va2 - R.va1, ddof=1):.4f} (n = {len(R)}), simulated {sim.std(ddof=1):.4f} (rho = {E.RHO_RT})")
    import pickle   # recommended schedule from optimize_static.py (run before this script)
    rec = pickle.load(open(E.OUT + "results.pkl", "rb"))["sel"][("all", "ALL")]["x"] if os.path.exists(E.OUT + "results.pkl") else np.array([14, 8, 9, 13, 14, 2, 2, 1, 0], float)
    sched = {"recommended": rec.tolist(), "comparator": E.CMP.tolist(), "every5_m2": E.SECOND["every5_m2"], "third": [6, 6, 10, 10, 12, 12, 1, 2, 4]}
    for k, x in sched.items():
        x = np.array(x, float); a, b = E.evaluate(x, C)[0], objectives(x, C)
        print(f"{k:12s} vectorised {np.round(a, 4)}  child-by-child {np.round(b, 4)}  max |diff| {np.abs(a - b).max():.2e}")
