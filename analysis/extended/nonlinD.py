import sys,json,time; sys.path.insert(0,".")
from lib import *
import lib
d=load(); y=d.logE.values; n=len(d)
def kfold_folds(rng,nn,K):
    p=rng.permutation(nn); return [p[i::K] for i in range(K)]
class DesignD:
    """design for model kinds, with D-nonlinearity fitted on training data only"""
def design2(dd,kind,df=4,fit=None):
    fit=fit if fit is not None else dd
    if kind in("M0","M1"): return design(dd,kind,df,fit)
    S=design(dd,"M0",df,fit); ld=dd.logD.values; c=fit.logD.mean()
    if kind=="Mq": return np.column_stack([S,ld-c,(ld-c)**2])
    if kind=="Mn":   # natural cubic spline in logD, 3 df, knots from the fitting data
        B=Basis(fit.logD.values,3); return np.column_stack([S,B(ld)[:,1:]-0*B(ld)[:,1:]]) if False else np.column_stack([S,B(ld)])[:, :S.shape[1]+2] 
    raise ValueError
# note: cr(df=3) -1 gives 3 columns that include an implicit intercept-free basis; combined with S (also intercept-free spline) it is collinear in the constant, drop first column
def design2(dd,kind,df=4,fit=None):
    fit=fit if fit is not None else dd
    if kind in("M0","M1"): return design(dd,kind,df,fit)
    S=design(dd,"M0",df,fit); ld=dd.logD.values
    if kind=="Mq":
        c=fit.logD.mean(); return np.column_stack([S,ld-c,(ld-c)**2])
    if kind=="Mn":
        B=Basis(fit.logD.values,3); return np.column_stack([S,B(ld)[:,1:]])
def loo_refit(kind,df=4):
    e=np.zeros(n)
    for i in range(n):
        tr=np.setdiff1d(np.arange(n),[i]); dtr,dte=d.iloc[tr],d.iloc[[i]]
        b=ols(design2(dtr,kind,df,dtr),y[tr]); e[i]=y[i]-(design2(dte,kind,df,dtr)@b)[0]
    return e
E={k:loo_refit(k) for k in ("M0","M1","Mq","Mn")}
print({k:round(float(np.sqrt(np.mean(v**2))),4) for k,v in E.items()})
out={}
for k in("Mq","Mn"):
    for ref in ("M0","M1"):
        dv=E[ref]**2-E[k]**2; ci=boot_ci(dv,B=10000); cc=boot_ci(dv,groups=d.cl.values,B=4000)
        rel=100*(1-np.sqrt(np.mean(E[k]**2))/np.sqrt(np.mean(E[ref]**2)))
        print(f"{k} vs {ref}: dMSE {dv.mean():.4f} [{ci[0]:.4f},{ci[1]:.4f}] cluster [{cc[0]:.4f},{cc[1]:.4f}] rel {rel:+.2f}%")
        out[f"{k}_vs_{ref}"]=dict(dMSE=float(dv.mean()),ci=ci.tolist(),cl=cc.tolist(),rel=float(rel))
# nested CV for Mq vs M0 and vs M1
def fit_pred(dtr,dte,kind,df=4): return design2(dte,kind,df,dtr)@ols(design2(dtr,kind,df,dtr),dtr.logE.values)
def dl(dtr,dte,ka,kb): yte=dte.logE.values; return (yte-fit_pred(dtr,dte,ka))**2-(yte-fit_pred(dtr,dte,kb))**2
def ncv(ka,kb,K=10,R=60,seed=3):
    rng=np.random.default_rng(seed); a=[];b=[];cvm=[];sec=[]
    for r in range(R):
        folds=kfold_folds(rng,n,K); dall=np.zeros(n)
        for k in range(K):
            te=folds[k]; tr=np.concatenate([folds[j] for j in range(K) if j!=k]); dtr,dte=d.iloc[tr],d.iloc[te]
            dout=dl(dtr,dte,ka,kb); dall[te]=dout; din=[]
            for j in range(K):
                if j==k: continue
                it=folds[j]; itr=np.concatenate([folds[m] for m in range(K) if m not in(j,k)]); din.append(dl(d.iloc[itr],d.iloc[it],ka,kb))
            din=np.concatenate(din); a.append((din.mean()-dout.mean())**2); b.append(dout.var(ddof=1)/len(te))
        cvm.append(dall.mean()); sec.append(dall.std(ddof=1)/np.sqrt(n))
    mse=(K-1)/K*(np.mean(a)-np.mean(b)); secv=float(np.mean(sec)); se=float(np.sqrt(max(mse,0))); sec_=min(max(se,secv),np.sqrt(K)*secv)
    return dict(cv=float(np.mean(cvm)),naive_se=secv,ncv_se=sec_,infl=sec_/secv)
t=time.time()
for ka,kb in (("M0","Mq"),("M1","Mq")):
    r=ncv(ka,kb); out["ncv_"+ka+"_"+kb]=r
    base=0.3609**2 if ka=="M0" else 0.3567**2
    lo,hi=r["cv"]-1.96*r["ncv_se"],r["cv"]+1.96*r["ncv_se"]
    print(f"nested CV {ka} vs {kb}: cv dMSE {r['cv']:.4f}, SE naive {r['naive_se']:.4f} nested {r['ncv_se']:.4f} (x{r['infl']:.2f}) interval [{lo:.4f},{hi:.4f}]  {time.time()-t:.0f}s")
json.dump(out,open("nonlinD.json","w"),indent=1)
