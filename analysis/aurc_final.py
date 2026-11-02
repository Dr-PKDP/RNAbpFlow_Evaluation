import os,sys,json; sys.path.insert(0,".")
import numpy as np, pandas as pd, patsy, statsmodels.api as sm
from aurc_crossfitted import cross_fitted_combined_score
from aurc_comparison import aurc
from nested_spline_model import build_design
S=os.environ.get("RNA_DATA", "data").rstrip("/") + "/"
data=json.load(open(S+"small/dispersion_error_results_deduped.json"))
L=np.array([x["length"] for x in data],float); D=np.array([x["dispersion"] for x in data]); E=np.array([x["error"] for x in data])
cl=json.load(open(S+"small/grouped_cv_results.json"))["clusters"]; mp={t:i for i,c in enumerate(cl) for t in c}
g=np.array([mp[x["target"]] for x in data])
c1=cross_fitted_combined_score(L,D,E)
# cross-fitted Model 0 prediction (same procedure, no dispersion)
ll=np.log(L); ld=np.log(D); le=np.log(E); n=len(L)
bi=patsy.dmatrix("cr(log_length, df=4) - 1",pd.DataFrame({"log_length":ll}),return_type="dataframe").design_info
c0=np.zeros(n)
for i in range(n):
    tr=[j for j in range(n) if j!=i]
    Xtr=build_design(ll[tr],ld[tr],4,False,False,bi); m=sm.OLS(le[tr],Xtr).fit()
    Xte=build_design(ll[[i]],ld[[i]],4,False,False,bi).reindex(columns=Xtr.columns,fill_value=0.0); c0[i]=m.predict(Xte)[0]
A={"length (raw)":aurc(L,E),"dispersion":aurc(D,E),"combined (cross-fitted M1)":aurc(c1,E),"length (cross-fitted M0)":aurc(c0,E),"oracle":aurc(E,E)}
rng=np.random.default_rng(0); A["random (mean of 20000)"]=float(np.mean([aurc(rng.random(n),E) for _ in range(20000)]))
for k,v in A.items(): print(f"{k:30s} {v:.4f}")
rs=A["random (mean of 20000)"]; o=A["oracle"]
print("share of random-to-oracle gap closed: length %.1f%% dispersion %.1f%% combined %.1f%%"%tuple(100*(rs-A[k])/(rs-o) for k in("length (raw)","dispersion","combined (cross-fitted M1)")))
u=np.unique(g); mem=[np.where(g==k)[0] for k in u]
def boot(fa,fb,B=4000,cluster=True,seed=1):
    r=np.random.default_rng(seed); out=[]
    for _ in range(B):
        idx=np.concatenate([mem[k] for k in r.integers(0,len(u),len(u))]) if cluster else r.integers(0,n,n)
        out.append(aurc(fa[idx],E[idx])-aurc(fb[idx],E[idx]))
    return np.percentile(out,[2.5,97.5])
res={}
for lab,fa,fb in [("length-combined",L,c1),("length-dispersion",L,D),("combinedM1-dispersion",c1,D),("M0cf-combined",c0,c1)]:
    t=boot(fa,fb,cluster=False); c=boot(fa,fb,cluster=True)
    d=aurc(fa,E)-aurc(fb,E)
    print(f"{lab}: diff {d:.3f} target-boot [{t[0]:.2f},{t[1]:.2f}] cluster-boot [{c[0]:.2f},{c[1]:.2f}]")
    res[lab]=dict(diff=float(d),t=t.tolist(),c=c.tolist())
json.dump(dict(A=A,res=res),open("aurc_final.json","w"),indent=1)
