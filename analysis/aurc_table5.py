import os,sys,json; sys.path.insert(0,".")
import numpy as np, pandas as pd, patsy, statsmodels.api as sm
from aurc_crossfitted import cross_fitted_combined_score
from nested_spline_model import build_design
from aurc_comparison import _trapezoidal
S=os.environ.get("RNA_DATA", "data").rstrip("/") + "/"
data=json.load(open(S+"small/dispersion_error_results_deduped.json"))
L=np.array([x["length"] for x in data],float); D=np.array([x["dispersion"] for x in data]); E=np.array([x["error"] for x in data])
cl=json.load(open(S+"small/grouped_cv_results.json"))["clusters"]; mp={t:i for i,c in enumerate(cl) for t in c}
g=np.array([mp[x["target"]] for x in data]); n=len(L)
c1=cross_fitted_combined_score(L,D,E)
ll=np.log(L); ld=np.log(D); le=np.log(E)
bi=patsy.dmatrix("cr(log_length, df=4) - 1",pd.DataFrame({"log_length":ll}),return_type="dataframe").design_info
c0=np.zeros(n)
for i in range(n):
    tr=[j for j in range(n) if j!=i]
    Xtr=build_design(ll[tr],ld[tr],4,False,False,bi); m=sm.OLS(le[tr],Xtr).fit()
    Xte=build_design(ll[[i]],ld[[i]],4,False,False,bi).reindex(columns=Xtr.columns,fill_value=0.0); c0[i]=m.predict(Xte)[0]
def tmean(x,f=0.1):
    x=np.sort(x); k=int(np.floor(len(x)*f)); return x.mean() if k==0 else x[k:len(x)-k].mean()
RISK={"mean":np.mean,"median":np.median,"trimmed":tmean}
FR=np.linspace(0.1,1.0,19)
def aurc(score,err,rf):
    o=np.argsort(score); N=len(err)
    cur=np.array([rf(err[o[:max(1,int(round(N*f)))]]) for f in FR])
    return _trapezoidal(cur,FR)/(FR[-1]-FR[0])
P={"length":L,"dispersion":D,"M0cf":c0,"M1cf":c1}
out={"A":{}, "res":{}}
rng=np.random.default_rng(0)
for rn,rf in RISK.items():
    out["A"][rn]={k:float(aurc(v,E,rf)) for k,v in P.items()}
    out["A"][rn]["oracle"]=float(aurc(E,E,rf))
    out["A"][rn]["random"]=float(np.mean([aurc(rng.random(n),E,rf) for _ in range(20000)]))
    print(rn,{k:round(v,2) for k,v in out["A"][rn].items()})
u=np.unique(g); mem=[np.where(g==k)[0] for k in u]
def boot(a,b,rf,cluster,B):
    r=np.random.default_rng(1 if cluster else 0); o=[]
    for _ in range(B):
        idx=np.concatenate([mem[k] for k in r.integers(0,len(u),len(u))]) if cluster else r.integers(0,n,n)
        o.append(aurc(a[idx],E[idx],rf)-aurc(b[idx],E[idx],rf))
    return np.percentile(o,[2.5,97.5]).tolist()
for rn,rf in RISK.items():
    for lab,a,b in [("length-M1cf",L,c1),("length-dispersion",L,D),("M0cf-M1cf",c0,c1)]:
        d=float(aurc(a,E,rf)-aurc(b,E,rf))
        t=boot(a,b,rf,False,2000); c=boot(a,b,rf,True,4000)
        out["res"][f"{rn}|{lab}"]=dict(diff=d,t=t,c=c)
        print(rn,lab,round(d,2),np.round(t,2),np.round(c,2))
json.dump(out,open("aurc_table5.json","w"),indent=1)
