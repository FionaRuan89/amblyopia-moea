"""Step 2: harmonise amblyopic-eye VA trajectories (time from start of current treatment phase).
Outputs (derived/):
  traj.csv   trial, pid, wk, va (logMAR), chart (snellen/hotv/etdrs)
  pts.csv    trial, pid, role (main/replay/aux), tx (spectacles/patching/atropine), age, va0, n_obs, last_wk
  retest.csv trial, chart, va1, va2   (same-day test/retest pairs, both eyes)
Main (treat-until-stable): ATS5 spectacle phase, ATS13 spectacles, ATS15 run-in (2 h patching), ATS16 run-in (weekend atropine).
Replay (model-free check): ATS12 (weekly VA).  Aux (fixed sparse visits): ATS1, ATS2A, ATS2B, ATS3, ATS6, ATS10
(read from derived/aux_*.csv written by harmonize_aux.py).  usage: python build_trajectories.py <root>
"""
import sys, glob, os, numpy as np, pandas as pd
ROOT = sys.argv[1].rstrip("/") + "/"; RAW = ROOT + "raw/"; OUT = ROOT + "derived/"; os.makedirs(OUT, exist_ok=True)

def rd(p):
    h = open(p, encoding="latin-1").read(4000); d = max(["|", ",", "\t"], key=h.count)
    df = pd.read_csv(p, sep=d, encoding="latin-1", low_memory=False); df.columns = [c.strip() for c in df.columns]; return df
def g(t, n): return rd(glob.glob(f"{RAW}{t}/extracted/**/{n}", recursive=True)[0])
def snel(s):
    def f(x):
        try: a, b = str(x).strip().split("/"); return np.log10(float(b) / float(a))
        except Exception: return np.nan
    return pd.Series(s).map(f).values
def letters(L): return ((85 - pd.to_numeric(pd.Series(L), errors="coerce")) / 50).values
def he(df, eye, pre=""):
    h = snel(df[f"{pre}HOTVVisAcu{eye}"]) if f"{pre}HOTVVisAcu{eye}" in df else np.full(len(df), np.nan)
    e = letters(df[f"{pre}ETDRSVisAcu{eye}"]) if f"{pre}ETDRSVisAcu{eye}" in df else np.full(len(df), np.nan)
    v = np.where(np.isnan(h), e, h); c = np.where(np.isnan(h), np.where(np.isnan(e), None, "etdrs"), "hotv")
    return v, c

TR, PT, RT = [], [], []
def add(trial, role, tx, pid, wk, va, chart, age):
    d = pd.DataFrame({"trial": trial, "pid": pid, "wk": wk, "va": va, "chart": chart}).dropna(subset=["wk", "va"])
    d = d[d.wk >= 0].sort_values(["pid", "wk"]).drop_duplicates(["pid", "wk"])
    TR.append(d)
    a = pd.Series(age).groupby(level=0).first() if isinstance(age, pd.Series) else None
    s = d.groupby("pid").agg(va0=("va", "first"), n_obs=("va", "size"), last_wk=("wk", "max"), first_wk=("wk", "min")).reset_index()
    s.insert(0, "trial", trial); s["role"] = role; s["tx"] = tx
    s["age"] = s.pid.map(a) if a is not None else np.nan
    PT.append(s)
def add_rt(trial, a, b, chart):
    ok = ~np.isnan(a) & ~np.isnan(b); RT.append(pd.DataFrame({"trial": trial, "chart": np.asarray(chart)[ok], "va1": a[ok], "va2": b[ok]}))

# ---------- ATS5 spectacle phase ----------
sb, fu = g("ATS05", "tblATS5SpecBaseline.csv"), g("ATS05", "tblATS5SpecFUVisit.csv")
ro = g("ATS05", "tblATS5PtRoster.csv").set_index("PtID")
for d in (sb, fu):
    d["dt"] = pd.to_datetime(d.VisitDt)
    for e in ("OD", "OS"): d["va" + e] = pd.Series(snel(d["FinalVA" + e])).fillna(pd.Series(snel(d["VisAcu" + e]))).values
eye = ro.StudyEye.reindex(sb.PtID).values
eye = pd.Series(np.where(pd.isna(eye), np.where(sb.vaOD.values >= sb.vaOS.values, "OD", "OS"), eye), index=sb.PtID.values)
t0 = sb.set_index("PtID").dt
fu = fu[fu.PtID.isin(eye.index)].sort_values(["PtID", "dt"]).copy()
fu = fu[~(fu.AmbTrt.astype(str) == "Yes").groupby(fu.PtID).cummax()]
d = pd.concat([sb[["PtID", "dt", "vaOD", "vaOS"]], fu[["PtID", "dt", "vaOD", "vaOS"]]])
e = d.PtID.map(eye)
add("ATS5", "main", "spectacles", d.PtID.values, ((d.dt - d.PtID.map(t0)).dt.days / 7).values,
    np.where(e == "OD", d.vaOD, d.vaOS), "snellen", ro.AgeAtBaseline)
for x in ("OD", "OS"): add_rt("ATS5", snel(fu["VisAcu" + x]), snel(fu["ReVisAcu" + x]), np.full(len(fu), "snellen"))

# ---------- ATS13 spectacles (every 9 wk until stable); stop after amblyopia treatment prescribed ----------
va = g("ATS13", "ATS13 Visual Acuity.txt"); ro = g("ATS13", "ATS13 Patient Roster.txt").set_index("PtID")
rx = g("ATS13", "ATS13 Treatment Prescribed - Follow up.txt")
cut = rx[rx.RxTreat == "Yes"].groupby("PtID").VisitDaysFromBaseline.min()
va = va[va.Visit != "Screening"].copy()
va = va[~(va.VATestDaysFromBaseline > va.PtID.map(cut).fillna(1e9) + 3)]
e = va.PtID.map(ro.StudyEye)
vod, cod = he(va, "OD"); vos, cos = he(va, "OS")
add("ATS13", "main", "spectacles", va.PtID.values, (va.VATestDaysFromBaseline / 7).values,
    np.where(e == "OD", vod, vos), np.where(e == "OD", cod, cos), ro.AgeAsOfBaseline)
for x in ("OD", "OS"):
    a, c = he(va, x); b, _ = he(va, x, "Re"); add_rt("ATS13", a, b, c)

# ---------- ATS15 / ATS16 run-in (time from current-treatment start) ----------
for t, f, tx, agecol in (("ATS15", "tblATS15", "patching", None), ("ATS16", "tblATS16", "atropine", "AgeAtEnroll")):
    va = g(t, f + "VisualAcuity.txt"); el = g(t, f + "EligCheck.txt").set_index("PtID"); ro = g(t, f + "PtRoster.txt").set_index("PtID")
    keep = el.index[el.EnrDisposition.isin(["No current treatment", "On treatment"])]
    va = va[va.Visit.isin(["Enrollment", "Follow-Up-Run-In"]) & va.PtID.isin(keep)].copy()
    vod, cod = he(va, "OD"); vos, cos = he(va, "OS")
    enr = va.Visit.values == "Enrollment"
    if "StudyEye" in ro: se = va.PtID.map(ro.StudyEye)
    else: se = pd.Series(np.nan, index=va.index)
    w = pd.Series(np.where(vod >= vos, "OD", "OS"), index=va.index)[enr]; w.index = va.PtID[enr].values
    se = se.fillna(va.PtID.map(w))
    off = el.DaysFromEnrollToCurrTreatStart.reindex(keep).fillna(0).clip(upper=0)
    wk = ((va.DaysFromEnrollToVATest - va.PtID.map(off)) / 7).values
    vv, cc = np.where(se == "OD", vod, vos), np.where(se == "OD", cod, cos)
    st = el.loc[keep]; stv = snel(st.CurrTreatStartVA); ok = ~np.isnan(stv) & (off.values < 0)   # VA at treatment start
    pid = np.concatenate([va.PtID.values, st.index[ok].values]); wk = np.concatenate([wk, np.zeros(ok.sum())])
    vv = np.concatenate([vv, stv[ok]]); cc = np.concatenate([cc, np.full(ok.sum(), "hotv")])
    add(t, "main", tx, pid, wk, vv, cc, ro[agecol] if agecol else np.nan)
    for x in ("OD", "OS"):
        a, c = he(va, x); b, _ = he(va, x, "Re"); add_rt(t, a, b, c)

# ---------- ATS12 weekly (replay) ----------
va = g("ATS12", "ATS12 Visual Acuity.txt"); ro = g("ATS12", "ATS12 Patient Roster.txt").set_index("PtID")
e = va.PtID.map(ro.StudyEye); vod, cod = he(va, "OD"); vos, cos = he(va, "OS")
add("ATS12", "replay", "patching", va.PtID.values, (va.VATestDaysFromEnroll / 7).values,
    np.where(e == "OD", vod, vos), np.where(e == "OD", cod, cos), ro.AgeAsOfEnrollmentDt)
for x in ("OD", "OS"):
    a, c = he(va, x); b, _ = he(va, x, "Re"); add_rt("ATS12", a, b, c)

# ---------- aux: harmonised fixed-visit trials (harmonize_aux.py) ----------
P = pd.read_csv(ROOT + "derived/aux_patients.csv"); V = pd.read_csv(ROOT + "derived/aux_visits.csv")
for t in ("ATS1", "ATS2A", "ATS2B", "ATS3", "ATS6", "ATS10"):
    p = P[P.trial == t].set_index("pid"); v = V[V.trial == t]
    for txc, pp in p.groupby("tx_class"):
        if txc not in ("patching", "atropine", "spectacles"): continue   # ATS3 active arm and ATS10 Bangerter arm are not used
        vv = v[v.pid.isin(pp.index)]
        add(t, "aux", txc, np.concatenate([pp.index.values, vv.pid.values]), np.concatenate([np.zeros(len(pp)), vv.week.values]),
            np.concatenate([pp.amb0.values, vv.amb.values]), "mixed", pp.age)

T, Pt, R = pd.concat(TR), pd.concat(PT), pd.concat(RT)
Pt = Pt[Pt.n_obs >= 2]; T = T.merge(Pt[["trial", "pid"]], on=["trial", "pid"])
T.to_csv(OUT + "traj.csv", index=False); Pt.to_csv(OUT + "pts.csv", index=False); R.to_csv(OUT + "retest.csv", index=False)
pd.set_option("display.width", 200)
print("=== patients per trial (>=2 VA obs) ===")
print(Pt.groupby(["role", "trial", "tx"]).agg(n=("pid", "size"), obs_med=("n_obs", "median"), obs_max=("n_obs", "max"),
      fu_wk_med=("last_wk", "median"), age_med=("age", "median"), age_missing=("age", lambda x: x.isna().mean()),
      va0_mean=("va0", "mean")).round(2).to_string())
print("\n=== visit-time distribution (weeks, rounded) in main trials ===")
for t, d in T[T.trial.isin(["ATS5", "ATS13", "ATS15", "ATS16"])].groupby("trial"):
    print(t, d.wk.round().astype(int).value_counts().sort_index().loc[lambda s: s >= 5].to_dict())
print("\n=== same-day retests ===")
R["diff"] = R.va2 - R.va1
print(R.groupby(["trial", "chart"])["diff"].agg(["size", "mean", "std"]).assign(per_test_sd=lambda x: x["std"] / np.sqrt(2)).round(3).to_string())
print("\nchart mix in main trajectories:", T[T.trial.isin(["ATS5", "ATS13", "ATS15", "ATS16"])].chart.value_counts().to_dict())
