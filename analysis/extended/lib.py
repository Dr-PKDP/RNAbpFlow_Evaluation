import os, json, numpy as np, pandas as pd
from patsy import dmatrix, build_design_matrices
SM=(os.environ.get("RNA_DATA", "data").rstrip("/") + "/small/")
def load():
    d=pd.DataFrame(json.load(open(SM+"dispersion_error_results_deduped.json")))
    d["logE"]=np.log(d.error); d["logL"]=np.log(d.length); d["logD"]=np.log(d.dispersion)
    cl=json.load(open(SM+"grouped_cv_results.json"))["clusters"]
    m={t:i for i,c in enumerate(cl) for t in c}
    d["cl"]=d.target.map(m)
    return d
class Basis:
    """natural cubic spline cr(df)-1 with knots from the fitting data"""
    def __init__(s,x,df):
        s.di=dmatrix(f"cr(x, df={df}) - 1",{"x":x}).design_info
        s.df=df
    def __call__(s,x): return np.asarray(build_design_matrices([s.di],{"x":x})[0])
def design(d,kind,df=4,fit=None):
    """kind: 'M0','M1' ,'L0' (linear logL),'L1','D' (logD only), 'Mvc'(spline+logD+logD*logL)"""
    x=d.logL.values
    fit = fit if fit is not None else d
    B=Basis(fit.logL.values,df)
    S=B(x)
    one=np.ones((len(d),1))
    if kind=="M0": return S
    if kind=="M1": return np.column_stack([S,d.logD.values])
    if kind=="Mvc":
        c=fit.logL.mean()
        return np.column_stack([S,d.logD.values,d.logD.values*(d.logL.values-c)])
    if kind=="L0": return np.column_stack([one,x])
    if kind=="L1": return np.column_stack([one,x,d.logD.values])
    if kind=="D": return np.column_stack([one,d.logD.values])
    raise ValueError(kind)
def ols(X,y):
    return np.linalg.lstsq(X,y,rcond=None)[0]
def loo_fixed(X,y):
    H=X@np.linalg.pinv(X.T@X)@X.T
    return (y-H@y)/(1-np.diag(H))
def cv_refit(d,kind,df=4,groups=None,folds=None):
    """leave-group-out with basis refit inside each training set; returns residual per target"""
    n=len(d); e=np.zeros(n)
    if folds is None:
        g=np.arange(n) if groups is None else np.asarray(groups)
        folds=[np.where(g==u)[0] for u in np.unique(g)]
    y=d.logE.values
    for te in folds:
        tr=np.setdiff1d(np.arange(n),te)
        dtr,dte=d.iloc[tr],d.iloc[te]
        Xtr=design(dtr,kind,df,fit=dtr); Xte=design(dte,kind,df,fit=dtr)
        b=ols(Xtr,y[tr]); e[te]=y[te]-Xte@b
    return e
def summ(e0,e1):
    d=e0**2-e1**2
    r0,r1=np.sqrt(np.mean(e0**2)),np.sqrt(np.mean(e1**2))
    return dict(rmse0=r0,rmse1=r1,rel=100*(1-r1/r0),dMSE=d.mean())
def boot_ci(dvec,groups=None,B=10000,seed=1):
    rng=np.random.default_rng(seed); n=len(dvec)
    if groups is None:
        idx=rng.integers(0,n,(B,n)); return np.percentile(dvec[idx].mean(1),[2.5,97.5])
    g=np.asarray(groups); u=np.unique(g); mem=[np.where(g==k)[0] for k in u]
    out=np.empty(B)
    for b in range(B):
        pick=rng.integers(0,len(u),len(u))
        out[b]=np.concatenate([dvec[mem[k]] for k in pick]).mean()
    return np.percentile(out,[2.5,97.5])
