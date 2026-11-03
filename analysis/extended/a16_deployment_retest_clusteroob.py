"""Deployment-compatible length/dispersion combinations, length-adjusted test-retest of log D,
cluster out-of-bootstrap refit, E versus D/sqrt(2) summary, and interval conversions.
Runs from the directory holding lib.py, bund.py, pairwise.npz and a10_table.json."""
import sys,json; sys.path.insert(0,".")
from lib import *
d=load(); a10=pd.DataFrame(json.load(open("a10_table.json"))).set_index("target")
P=np.load("pairwise.npz"); T=d.target.tolist()
# ---- (1) E vs D/sqrt2 and D<=2E with primary values
r=(d.error/(d.dispersion/np.sqrt(2)))
print("E/(D/sqrt2): min %.3f (%s), median E/D %.3f ; max D/E %.3f"%(r.min(),d.target[r.idxmin()],np.median(d.error/d.dispersion),(d.dispersion/d.error).max()))
# ---- (2) deployment-compatible combinations
def gain(L,D,E,idx=None):
    dd=pd.DataFrame(dict(logL=np.log(L),logD=np.log(D),logE=np.log(E)));yy=dd.logE.values
    e0=loo_fixed(design(dd,"M0"),yy);e1=loo_fixed(design(dd,"M1"),yy);s=summ(e0,e1);ci=boot_ci(e0**2-e1**2,B=10000)
    return s,ci
a=a10.loc[T]
for lab,(L,D) in {"resolved L, resolved D (primary)":(a.n_paired,a.D_res),"resolved L, full D":(a.n_paired,a.D_all),"full predicted L, resolved D":(a.n_pred,a.D_res),"full predicted L, full D (deployment-compatible)":(a.n_pred,a.D_all)}.items():
    s,ci=gain(L.values,D.values,a.error.values)
    print("%-52s rmse0 %.4f rmse1 %.4f rel %.2f%% dMSE %.4f [%.4f,%.4f]  rel-int [%.1f,%.1f]"%(lab,s['rmse0'],s['rmse1'],s['rel'],s['dMSE'],ci[0],ci[1],100*(1-np.sqrt(1-ci[0]/s['rmse0']**2)),100*(1-np.sqrt(1-ci[1]/s['rmse0']**2))))
# ---- (3) length-adjusted test-retest of log D (primary vs dropout-off vs extra10)
M={t:P[f"{t}__M"] for t in T}
def meanoff(Mm,idx): s=Mm[np.ix_(idx,idx)];return s[np.triu_indices(len(idx),1)].mean()
ix={"p":np.arange(20),"o":np.arange(20,40),"x":np.arange(40,50)}
lD={k:np.log([meanoff(M[t],v) for t in T]) for k,v in ix.items()}
S=design(d,"M0")
def resid(v): return v-S@ols(S,v)
R={k:resid(v) for k,v in lD.items()}
for x,y in (("p","o"),("p","x"),("o","x")):
    print("test-retest log D %s-%s raw r=%.3f ; length-adjusted r=%.3f ; Spearman adj %.3f"%(x,y,np.corrcoef(lD[x],lD[y])[0,1],np.corrcoef(R[x],R[y])[0,1],pd.Series(R[x]).corr(pd.Series(R[y]),method='spearman')))
print("SD of residualised logD: %.3f ; SD logD %.3f"%(R['p'].std(),lD['p'].std()))
# ---- (4) cluster out-of-bootstrap refit
rng=np.random.default_rng(5); cl=d.cl.values; u=np.unique(cl); mem={k:np.where(cl==k)[0] for k in u}
out=[];nb=0
for b in range(4000):
    pick=rng.choice(u,len(u),replace=True); inb=np.concatenate([mem[k] for k in pick]); oob=np.setdiff1d(np.arange(len(d)),inb)
    if len(oob)<8 or len(np.unique(d.logL.values[inb]))<8: continue
    dtr,dte=d.iloc[inb],d.iloc[oob]
    try:
        y=dtr.logE.values;yt=dte.logE.values
        X0=design(dtr,"M0",fit=dtr);X1=design(dtr,"M1",fit=dtr)
        e0=yt-design(dte,"M0",fit=dtr)@ols(X0,y);e1=yt-design(dte,"M1",fit=dtr)@ols(X1,y)
    except Exception as ex: continue
    out.append(100*(1-np.sqrt(np.mean(e1**2))/np.sqrt(np.mean(e0**2))))
out=np.array(out);print("cluster OOB: n=%d mean %.2f sd %.2f 95%% [%.1f,%.1f] frac<0 %.3f"%(len(out),out.mean(),out.std(),*np.percentile(out,[2.5,97.5]),(out<0).mean()))
# ---- (5) conversions
base=0.3609**2
for lab,(lo,hi) in {"cluster":(-0.0170,0.0245),"primary":(-0.0156,0.0221),"nested exact":(0.0037934-1.96*0.0182402,0.0037934+1.96*0.0182402)}.items():
    print(lab,"%.1f %.1f"%tuple(100*(1-np.sqrt(1-x/base)) for x in (lo,hi)))
