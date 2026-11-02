import sys,time; sys.path.insert(0,".")
from lib import *; from bund import get,gt; from metrics import *
d=load(); T=d.target.tolist(); t0=time.time(); rows=[]
for t in T:
    X=get(t,"primary"); G=gt(t); tm=[tm_score(x,G) for x in X]; ld=[lddt(x,G) for x in X]; cf=[contact_f1(x,G) for x in X]
    rows.append(dict(target=t,tm=np.mean(tm),lddt=np.mean(ld),f1=np.mean(cf)))
m=pd.DataFrame(rows); m.to_json("a11_metrics.json",orient="records"); print("time %.0fs"%(time.time()-t0))
d=d.merge(m,on="target")
print(m[["tm","lddt","f1"]].describe().round(3).loc[["mean","50%","min","max"]])
print("Spearman with RMSD error:", {k:round(d[[k,"error"]].corr("spearman").iloc[0,1],3) for k in ("tm","lddt","f1")})
for lab,col in (("TM-score (C4', RNA d0)","tm"),("lDDT-C4'","lddt"),("contact F1 (12 A)","f1")):
    yy=np.log(np.clip(1-d[col].values,1e-3,None)) if col!="f1" else np.log(np.clip(1-d[col].values,1e-3,None))
    dd=d.assign(logE=yy); e0=loo_fixed(design(dd,"M0"),yy); e1=loo_fixed(design(dd,"M1"),yy); s=summ(e0,e1); ci=boot_ci(e0**2-e1**2); cc=boot_ci(e0**2-e1**2,d.cl.values)
    b=ols(design(dd,"M1"),yy)[-1]
    print(f"  outcome log(1-{lab:24s}) M0 {s['rmse0']:.3f} M1 {s['rmse1']:.3f} rel {s['rel']:6.2f}% dMSE {s['dMSE']:.4f} [{ci[0]:.4f},{ci[1]:.4f}] cluster [{cc[0]:.4f},{cc[1]:.4f}] coef {b:.3f}")
# raw-scale outcome E in Å without log, linear in D
y=d.error.values; e0=loo_fixed(design(d,"M0"),y); X=np.column_stack([design(d,"M0"),d.dispersion.values]); e1=loo_fixed(X,y); s=summ(e0,e1); print("  raw-Å outcome (spline L + D linear): rmse0 %.2f rmse1 %.2f rel %.2f%%"%(s["rmse0"],s["rmse1"],s["rel"]))
