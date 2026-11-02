import sys,json; sys.path.insert(0,".")
from lib import *
from collections import Counter
F=json.load(open(SM+"../rf/rfam_scan/families.json"))
d=load(); T=d.target.tolist()
trainTop=Counter(F[k]["top"]["acc"] for k in F if k.startswith("R:") and F[k]["top"])
trainAll=Counter(a for k in F if k.startswith("R:") for a in F[k]["families"])
print("training chains with hit: %d/1204; distinct top families %d; distinct families (any hit) %d"%(sum(1 for k in F if k.startswith('R:') and F[k]['top']),len(trainTop),len(trainAll)))
rows=[]
for t in T:
    f=F[f"T:{t}"]; top=f["top"]
    rows.append(dict(target=t,fam=top["acc"] if top else None,model=top["model"] if top else None,cov=top["coverage"] if top else 0,ev=top["evalue"] if top else None,
                     in_train_top=bool(top and top["acc"] in trainTop),in_train_any=bool(f["families"] and any(a in trainAll for a in f["families"])),n_train_top=trainTop.get(top["acc"],0) if top else 0))
r=pd.DataFrame(rows); d=d.merge(r,on="target")
print("targets with Rfam hit %d/82; family present in training (top-hit) %d; (any hit) %d; no Rfam hit %d; hit but family absent in training %d"%(d.fam.notna().sum(),d.in_train_top.sum(),d.in_train_any.sum(),d.fam.isna().sum(),(d.fam.notna()&~d.in_train_top).sum()))
print(d.groupby("model").agg(n=("target","size"),n_train=("n_train_top","first"),med_len=("length","median")).sort_values("n",ascending=False).head(12).to_string())
print("absent-family targets:",d[d.fam.notna()&~d.in_train_top][["target","model","length"]].values.tolist())
def g(dd,lab):
    dd=dd.reset_index(drop=True); y=dd.logE.values
    if len(dd)<12: print(f"  {lab}: N={len(dd)} (too few)"); return
    e0=loo_fixed(design(dd,"M0"),y); e1=loo_fixed(design(dd,"M1"),y); s=summ(e0,e1); ci=boot_ci(e0**2-e1**2)
    print(f"  {lab:46s} N={len(dd):2d} M0 {s['rmse0']:.3f} M1 {s['rmse1']:.3f} rel {s['rel']:6.2f}% dMSE {s['dMSE']:7.4f} [{ci[0]:.4f},{ci[1]:.4f}]")
print("Subsets (own-fit LOOCV, refit knots on subset):")
g(d,"all 82"); g(d[~d.in_train_top],"family not in training (incl. no hit)"); g(d[d.fam.isna()],"no Rfam hit"); g(d[d.fam.notna()&~d.in_train_top],"Rfam hit, family absent from training")
g(d[d.in_train_top],"family present in training")
# leave-family-out CV: groups = Rfam top family; no-hit targets grouped by their sequence cluster
grp=np.where(d.fam.notna(),"F_"+d.fam.fillna(""),"S_"+d.cl.astype(str)); gid=pd.factorize(grp)[0]; print("leave-family-out groups:",len(np.unique(gid)),"largest",np.bincount(gid).max())
# merge with sequence clusters transitively? keep simple
y=d.logE.values; X0,X1=design(d,"M0"),design(d,"M1")
def lgo(X):
    e=np.zeros(len(d))
    for c in np.unique(gid):
        te=np.where(gid==c)[0]; tr=np.setdiff1d(np.arange(len(d)),te); e[te]=y[te]-X[te]@ols(X[tr],y[tr])
    return e
a,b=lgo(X0),lgo(X1); s=summ(a,b); print("  leave-family-out: M0 %.4f M1 %.4f rel %.2f%% dMSE %.4f cluster-CI %s"%(s["rmse0"],s["rmse1"],s["rel"],s["dMSE"],boot_ci(a**2-b**2,gid).round(4)))
d.to_json("c1_table.json",orient="records")
# overlap with the identity-based flags
to=pd.DataFrame(json.load(open(SM+"training_overlap_detailed.json")))[["target","global_percent_identity","excluded_at_80pct_threshold"]]
m=d.merge(to,on="target"); print("family-in-training vs identity>=0.8 flag crosstab:"); print(pd.crosstab(m.in_train_top,m.excluded_at_80pct_threshold))
print("mean global identity by family-in-training:",m.groupby("in_train_top").global_percent_identity.mean().round(3).to_dict())
