import os,sys,json; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib import *
C2=os.environ.get("RNA_DATA", "data").rstrip("/") + "/c2/"
rows=[json.loads(l) for l in open(C2+"null_full.json.jsonl")]
R=pd.DataFrame(rows); print(len(R),"rows;",R.groupby("kind").size().to_dict())
to=pd.DataFrame(json.load(open(SM+"training_overlap_detailed.json"))).set_index("target")
real=R[R.kind=="real"].set_index("target"); nul=R[R.kind!="real"].copy()
j=real.join(to,how="inner"); print("real reproduced:",len(j))
print("max |global_id - reported| = %.4f ; local_id %.4f ; cov %.4f"%((j.global_id-j.global_percent_identity).abs().max(),(j.local_id-j.local_percent_identity).abs().max(),(j.local_cov_target-j.local_coverage_of_target).abs().max()))
# length bins by resolved sequence length
bins=[0,30,60,120,10**6]; labs=["<=30","31-60","61-120",">120"]
for df_ in (R,nul,real): pass
R["bin"]=pd.cut(R.qlen,bins,labels=labs); nul["bin"]=pd.cut(nul.qlen,bins,labels=labs); real["bin"]=pd.cut(real.qlen,bins,labels=labs)
nul["loc_flag"]=(nul.local_id>=.93)&(nul.local_cov_target>=.5)
nul["glob_flag"]=nul.global_id>=.80
print("\nNull (shuffled) distribution of max global identity by length bin")
tab=[]
for kind in ("dinuc","mono"):
    for b in labs:
        s=nul[(nul.kind==kind)&(nul.bin==b)]
        if len(s)==0: continue
        tab.append(dict(kind=kind,bin=b,n=len(s),med=s.global_id.median(),p95=s.global_id.quantile(.95),p99=s.global_id.quantile(.99),mx=s.global_id.max(),frac80=s.glob_flag.mean(),frac_local=s.loc_flag.mean(),loc_p95=s.local_id.quantile(.95)))
T=pd.DataFrame(tab); print(T.round(3).to_string(index=False))
# real vs null pooled by bin (dinuc+mono)
print("\nReal targets by bin: median global id, and n real >= null p95 / p99 of the same bin (dinuc)")
thr={}
for b in labs:
    s=nul[(nul.kind=="dinuc")&(nul.bin==b)]
    if len(s): thr[b]=dict(p95=s.global_id.quantile(.95),p99=s.global_id.quantile(.99),lp95=s.local_id.quantile(.95),lp99=s.local_id.quantile(.99))
real["gthr95"]=real.bin.astype(str).map(lambda b:thr.get(b,{}).get("p95",np.nan)).astype(float); real["gthr99"]=real.bin.astype(str).map(lambda b:thr.get(b,{}).get("p99",np.nan)).astype(float)
real["cal95"]=real.global_id>real.gthr95; real["cal99"]=real.global_id>real.gthr99
# per-target z against its own shuffles
g=nul[nul.kind=="dinuc"].groupby("target").global_id.agg(["mean","std","max","count"])
real=real.join(g.rename(columns=lambda c:"n_"+c))
real["z"]=(real.global_id-real.n_mean)/real.n_std
real["above_own_max"]=real.global_id>real.n_max
real["flag80"]=real.global_id>=.80
real["flagloc"]=(~real.flag80)&(real.local_id>=.93)&(real.local_cov_target>=.5)
print(real.groupby("bin").agg(n=("qlen","size"),med_gid=("global_id","median"),flag80=("flag80","sum"),flagloc=("flagloc","sum"),cal95=("cal95","sum"),cal99=("cal99","sum"),above_own=("above_own_max","sum")).to_string())
print("totals: flag80",int(real.flag80.sum()),"flag local-only",int(real.flagloc.sum()),"cal95",int(real.cal95.sum()),"cal99",int(real.cal99.sum()),"above own max",int(real.above_own_max.sum()))
print("targets flagged by cal99 but not by 80/local:",sorted(real[real.cal99&~real.flag80&~real.flagloc].index))
print("targets flagged by 80 but not cal99:",sorted(real[real.flag80&~real.cal99].index))
# Table-4-style: exclude calibrated-flagged targets
d=load(); y=d.logE.values
def delta(sub):
    s=sub.reset_index(drop=True); yy=s.logE.values
    a,b=loo_fixed(design(s,"M0"),yy),loo_fixed(design(s,"M1"),yy); r=summ(a,b); ci=boot_ci(a**2-b**2,B=4000); r["lo"],r["hi"]=ci; r["n"]=len(s); return r
out={}
def run(name,excl):
    r=delta(d[~d.target.isin(excl)]); out[name]=r
    print(f"{name:42s} N={r['n']:3d} RMSE0 {r['rmse0']:.4f} RMSE1 {r['rmse1']:.4f} rel {r['rel']:+.2f}% dMSE {r['dMSE']:.4f} [{r['lo']:.4f},{r['hi']:.4f}]")
print("\nTable-4-style re-analysis")
run("all",[])
run("excl identity>=80% (reported)",list(real[real.flag80].index))
run("excl 80% + local flag (reported)",list(real[real.flag80|real.flagloc].index))
run("excl calibrated p95 (length bin)",list(real[real.cal95].index))
run("excl calibrated p99 (length bin)",list(real[real.cal99].index))
run("excl any of 80/local/cal99",list(real[real.flag80|real.flagloc|real.cal99].index))
real.drop(columns=[c for c in real.columns if c.endswith("_y")],errors="ignore").reset_index().to_json(C2+"c2_real_flags.json",orient="records")
json.dump(dict(null_table=T.to_dict("records"),thr=thr,refit=out),open(C2+"c2_summary.json","w"),indent=1,default=float)
