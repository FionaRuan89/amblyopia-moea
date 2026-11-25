"""Step 1: harmonise the auxiliary PEDIG trials (fixed, sparse visit schedules) into two long tables.
   derived/aux_patients.csv : trial, pid, site, arm, tx_class, age, amb0, fel0 (logMAR), adaptive, final_week
   derived/aux_visits.csv   : trial, pid, week, amb (logMAR), source
Trials: ATS1, ATS2A, ATS2B, ATS3, ATS6, ATS10.

VA conversion: Snellen 20/x -> log10(x/20); E-ETDRS letters -> (85 - L)/50; logMAR columns used as is.
usage: python harmonize_aux.py <root>      (raw data in <root>/raw/<STUDY>/extracted/)
"""
import glob, os, sys
import numpy as np, pandas as pd

ROOT = sys.argv[1].rstrip("/") + "/"
RAW = ROOT + "raw/"

def rd(p):
    h = open(p, encoding="latin-1").read(4000)
    d = max(["|", ",", "\t"], key=h.count)
    df = pd.read_csv(p, sep=d, encoding="latin-1", low_memory=False)
    df.columns = [c.strip().strip('"') for c in df.columns]
    return df

def g(trial, name):
    return rd(glob.glob(f"{RAW}{trial}/extracted/**/{name}", recursive=True)[0])

def snel(s):
    def f(x):
        try:
            a, b = str(x).strip().split("/"); return np.log10(float(b) / float(a))
        except Exception:
            return np.nan
    return pd.Series(s).map(f).values

def letters(L):
    return ((85 - pd.to_numeric(pd.Series(L), errors="coerce")) / 50).values

def hotv_etdrs(df, eye):
    v = pd.Series(snel(df[f"HOTVVisAcu{eye}"]), index=df.index)
    return v.fillna(pd.Series(letters(df[f"ETDRSVisAcu{eye}"]), index=df.index)).values

P, V = [], []

def add(trial, pts, visits):
    pts = pts.copy(); visits = visits.copy()
    pts.insert(0, "trial", trial); visits.insert(0, "trial", trial)
    P.append(pts); V.append(visits.dropna(subset=["amb"]))

# ---------------- ATS1 (wide; logMAR): 5 wk (A), 16 wk (B), 26 wk (6-month primary outcome)
d = g("ATS1", "ATS1NTIS.csv").reset_index().rename(columns={"index": "pid"})
pts = pd.DataFrame({"pid": d.pid, "site": np.nan, "arm": d.RxGroup,
                    "tx_class": d.RxGroup.map({"Occlusion": "patching", "Pharmacologic": "atropine"}),
                    "age": d.ageyrsrand, "amb0": d.amblog0, "fel0": d.sndlog0, "adaptive": 0, "final_week": 26})
vis = pd.concat([pd.DataFrame({"pid": d.pid, "week": w, "amb": d[c], "source": c})
                 for w, c in [(5, "ambloga"), (16, "amblogb"), (26, "amblog6")]])
add("ATS1", pts, vis)

# ---------------- ATS2A / ATS2B (wide; logMAR; actual days)
for t in ("ATS2A", "ATS2B"):
    d = g(t, f"{t}NTIS.csv").reset_index().rename(columns={"index": "pid"})
    pts = pd.DataFrame({"pid": d.pid, "site": np.nan, "arm": d.TRTGROUP, "tx_class": "patching",
                        "age": d.age, "amb0": d.amblog0, "fel0": d.sndlog0, "adaptive": 0, "final_week": 17})
    vis = pd.concat([pd.DataFrame({"pid": d.pid, "week": d.VISATIME / 7, "amb": d.ambloga, "source": "A"}),
                     pd.DataFrame({"pid": d.pid, "week": d.VISBTIME / 7, "amb": d.amblogb, "source": "B"})])
    add(t, pts, vis)

# ---------------- ATS3 (E-ETDRS letters; only the spectacles control arm is used downstream)
d3 = glob.glob(RAW + "ATS3/extracted/*/DataTables/")[0]
ro = rd(d3 + "tblATS3PtRoster.csv").set_index("PtID")
rnd = rd(d3 + "tblATS3Rand.csv").set_index("PtID").reindex(ro.index)
vi = rd(d3 + "tblATS3Visits.csv")
vi = vi[vi.Visit.fillna("").str.startswith("Randomized-")].copy()
vi["week"] = vi.Visit.str.extract(r"(\d+)wks")[0].astype(float)
od = ro.StudyEye.eq("OD")
pts = pd.DataFrame({"pid": ro.index, "site": ro.SiteID.values, "arm": ro.TrtGroup.values,
                    "tx_class": np.where(ro.TrtGroup.eq("Active"), "patching+atropine", "spectacles"),
                    "age": ro.AgeAtBaseline.values, "amb0": letters(rnd.VisAcuLtrAmb),
                    "fel0": np.where(od, letters(rnd.VisAcuLtrOS), letters(rnd.VisAcuLtrOD)),
                    "adaptive": 1, "final_week": np.nan})
vis = pd.DataFrame({"pid": vi.PtID.values, "week": vi.week.values, "amb": letters(vi.VisAcuLtrAmb), "source": vi.Visit.values})
add("ATS3", pts, vis)

# ---------------- ATS6 (Snellen; 2/5/8/17 wk)
ro, en, fu = g("ATS06", "TblATS06PtRoster.csv").set_index("PtID"), g("ATS06", "TblATS06Enroll.csv"), g("ATS06", "TblATS06FuVisit.csv")
en = en.drop_duplicates("PtID").set_index("PtID").reindex(ro.index); od = ro.StudyEye.eq("OD")
pts = pd.DataFrame({"pid": ro.index, "site": np.nan, "arm": ro.TrtGroup.values, "tx_class": "patching",
                    "age": ro.AgeAsOfEnrollmentDt.values, "amb0": np.where(od, snel(en.VisAcuOD), snel(en.VisAcuOS)),
                    "fel0": np.where(od, snel(en.VisAcuOS), snel(en.VisAcuOD)), "adaptive": 0, "final_week": 17})
fu = fu[fu.Visit.str.contains(r"\d+wks", na=False)].copy()
fu["week"] = fu.Visit.str.extract(r"(\d+)wks")[0].astype(float)
odv = fu.PtID.map(ro.StudyEye).eq("OD")
vis = pd.DataFrame({"pid": fu.PtID.values, "week": fu.week.values,
                    "amb": np.where(odv, snel(fu.VisAcuOD), snel(fu.VisAcuOS)), "source": fu.Visit.values})
add("ATS6", pts, vis)

# ---------------- ATS10 (HOTV / E-ETDRS; 6/12/18/24 wk; only the patching arm is used downstream)
ro, va = g("ATS10", "ATS10 Patient Roster.csv").set_index("PtID"), g("ATS10", "ATS10 Visual Acuity.csv")
va["ambv"] = np.where(va.PtID.map(ro.StudyEye).eq("OD"), hotv_etdrs(va, "OD"), hotv_etdrs(va, "OS"))
va["felv"] = np.where(va.PtID.map(ro.StudyEye).eq("OD"), hotv_etdrs(va, "OS"), hotv_etdrs(va, "OD"))
b = va[va.Visit.eq("Enrollment")].drop_duplicates("PtID").set_index("PtID").reindex(ro.index)
pts = pd.DataFrame({"pid": ro.index, "site": np.nan, "arm": ro.TrtGroup.values,
                    "tx_class": ro.TrtGroup.map({"Patching": "patching", "Bangerter": "bangerter"}).values,
                    "age": ro.AgeAsOfEnrollmentDt.values, "amb0": b.ambv.values, "fel0": b.felv.values,
                    "adaptive": 0, "final_week": 24})
f = va[va.Visit.str.contains(r"\d+wks", na=False)].copy(); f["week"] = f.Visit.str.extract(r"(\d+)wks")[0].astype(float)
add("ATS10", pts, pd.DataFrame({"pid": f.PtID.values, "week": f.week.values, "amb": f.ambv.values, "source": f.Visit.values}))

# ---------------- write
os.makedirs(ROOT + "derived", exist_ok=True)
DP, DV = pd.concat(P), pd.concat(V)
DP.to_csv(ROOT + "derived/aux_patients.csv", index=False)
DV.to_csv(ROOT + "derived/aux_visits.csv", index=False)
for t, p in DP.groupby("trial", sort=False):
    v = DV[DV.trial == t]
    print(f"{t:6s} n={len(p):4d} tx={p.tx_class.value_counts().to_dict()} | "
          f"patients with VA by week: {v.groupby(v.week.round()).pid.nunique().to_dict()}")
