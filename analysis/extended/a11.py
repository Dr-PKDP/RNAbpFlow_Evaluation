import sys,time; sys.path.insert(0,".")
from lib import *; from bund import get,gt,Z; from metrics import *
d=load(); T=d.target.tolist(); t0=time.time(); rows=[]
for t in T:
    X=get(t,"primary"); G=gt(t); tm=[tm_score(x,G) for x in X]; ld=[lddt(x,G) for x in X]; cf=[contact_f1(x,G,idx=Z[f"{t}__pred_idx"]) for x in X]
    rows.append(dict(target=t,tm=np.mean(tm),lddt=np.mean(ld),f1=np.mean(cf)))
m=pd.DataFrame(rows); m.to_json("a11_metrics.json",orient="records"); print("time %.0fs"%(time.time()-t0))
d=d.merge(m,on="target")
print(m[["tm","lddt","f1"]].describe().round(3).loc[["mean","50%","min","max"]])
print("Spearman with RMSD error:", {k:round(d[[k,"error"]].corr("spearman").iloc[0,1],3) for k in ("tm","lddt","f1")})
# outcomes of Table 8: log TM-score, logit lDDT, log(contact F1 + 0.02)
OUT=(("log TM-score (C4', RNA d0)",lambda v:np.log(v),"tm"),("logit lDDT-C4' (15 A)",lambda v:np.log(v/(1-v)),"lddt"),("log (contact F1 + 0.02), 12 A",lambda v:np.log(v+0.02),"f1"))
for lab,fn,col in OUT:
    yy=fn(d[col].values)
    dd=d.assign(logE=yy); e0=loo_fixed(design(dd,"M0"),yy); e1=loo_fixed(design(dd,"M1"),yy); s=summ(e0,e1); ci=boot_ci(e0**2-e1**2); cc=boot_ci(e0**2-e1**2,d.cl.values)
    b=ols(design(dd,"M1"),yy)[-1]
    print(f"  outcome {lab:32s} M0 {s['rmse0']:.3f} M1 {s['rmse1']:.3f} rel {s['rel']:6.2f}% dMSE {s['dMSE']:.4f} [{ci[0]:.4f},{ci[1]:.4f}] cluster [{cc[0]:.4f},{cc[1]:.4f}] coef {b:.3f}")
# raw-scale outcome E in Å without log, linear in D
y=d.error.values; e0=loo_fixed(design(d,"M0"),y); X=np.column_stack([design(d,"M0"),d.dispersion.values]); e1=loo_fixed(X,y); s=summ(e0,e1); print("  raw-Å outcome (spline L + D linear): rmse0 %.2f rmse1 %.2f rel %.2f%%"%(s["rmse0"],s["rmse1"],s["rel"]))
