import sys,json; sys.path.insert(0,".")
from lib import *
d=load(); y=d.logE.values
m=json.load(open(SM+"../bundle/meta.json"))
gc=lambda s:(s.count('G')+s.count('C'))/len(s)
d["gc"]=d.target.map(lambda t:gc(m[t]["gt_seq"]))
S=design(d,"M0"); D=d.logD.values[:,None]; g=d.gc.values[:,None]
X={"M0":S,"M1":np.column_stack([S,D]),"MG":np.column_stack([S,g]),"MGD":np.column_stack([S,g,D])}
E={k:loo_fixed(v,y) for k,v in X.items()}
print("N",len(d),{k:round(float(np.sqrt(np.mean(e**2))),4) for k,e in E.items()})
for a,b in (("M0","M1"),("MG","MGD"),("M0","MG"),("M0","MGD")):
    s=summ(E[a],E[b]); ci=boot_ci(E[a]**2-E[b]**2,B=10000); print(a,"->",b,"rel %.2f%% dMSE %.4f [%.4f,%.4f]"%(s["rel"],s["dMSE"],*ci))
print("corr(gc,logL) %.2f"%np.corrcoef(d.gc,d.logL)[0,1])
# repeated 5-fold CV mean for MG vs MGD (200 repeats)
import numpy as np
def rcv(Xa,Xb,R=200,K=5,seed=5):
    rng=np.random.default_rng(seed); rel=[]
    for r in range(R):
        p=rng.permutation(len(d)); fo=[p[i::K] for i in range(K)]; ea=np.zeros(len(d)); eb=np.zeros(len(d))
        for te in fo:
            tr=np.setdiff1d(np.arange(len(d)),te); dtr,dte=d.iloc[tr],d.iloc[te]
            def mk(kind):
                S1=design(dtr,"M0",4,dtr); S2=design(dte,"M0",4,dtr)
                if kind=="MG": return np.column_stack([S1,dtr.gc.values]),np.column_stack([S2,dte.gc.values])
                return np.column_stack([S1,dtr.gc.values,dtr.logD.values]),np.column_stack([S2,dte.gc.values,dte.logD.values])
            for kind,arr in(("MG",ea),("MGD",eb)):
                a,b=mk(kind); arr[te]=y[te]-b@ols(a,y[tr])
        rel.append(100*(1-np.sqrt(np.mean(eb**2))/np.sqrt(np.mean(ea**2))))
    return np.mean(rel),np.std(rel),np.mean(np.array(rel)<0)
print("repeated 5-fold (200) MG->MGD mean %.2f%% sd %.2f%% worse %.1f%%"%rcv(0,0))
# identity covariate N=82 using C2 real values
r=pd.DataFrame(json.load(open(SM+"../c2/c2_real_flags.json")))[["target","global_id"]]
d=d.merge(r,on="target"); idn=((d.global_id-d.global_id.mean())/d.global_id.std()).values
y=d.logE.values; S=design(d,"M0"); a=loo_fixed(S,y); b=loo_fixed(np.column_stack([S,idn]),y); c=loo_fixed(np.column_stack([S,idn,d.logD.values]),y)
print("identity N=%d: M0 %.4f +id %.4f (rel %.2f%%) +id+D %.4f (D|id rel %.2f%%, dMSE %.4f)"%(len(d),np.sqrt((a**2).mean()),np.sqrt((b**2).mean()),summ(a,b)["rel"],np.sqrt((c**2).mean()),summ(b,c)["rel"],summ(b,c)["dMSE"]))
