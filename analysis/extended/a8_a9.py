import sys,json; sys.path.insert(0,".")
from lib import *
from scipy.stats import spearmanr
P=np.load("pairwise.npz"); d=load(); T=d.target.tolist()
M={t:P[f"{t}__M"] for t in T}; Ev={t:P[f"{t}__E"] for t in T}
y_prim=None
def meanoff(Mm,idx):
    idx=np.asarray(idx); s=Mm[np.ix_(idx,idx)]; k=len(idx); return s[np.triu_indices(k,1)].mean()
ix={"primary":np.arange(20),"dropout_off":np.arange(20,40),"extra10":np.arange(40,50),"all":np.arange(50)}
def table(Dsrc,Esrc):
    D=np.array([meanoff(M[t],ix[Dsrc]) for t in T]); E=np.array([Ev[t][ix[Esrc]].mean() for t in T]); return D,E
def gain(D,E,L=d.logL.values):
    dd=pd.DataFrame(dict(logL=L,logD=np.log(D),logE=np.log(E))); yy=dd.logE.values
    e0=loo_fixed(design(dd,"M0"),yy); e1=loo_fixed(design(dd,"M1"),yy); s=summ(e0,e1); ci=boot_ci(e0**2-e1**2)
    b=ols(design(dd,"M1"),yy)[-1]; return s,ci,b
print("reliability of log D between independent ensembles:")
Dp,Ep=table("primary","primary"); Do,Eo=table("dropout_off","dropout_off"); Dx,Ex=table("extra10","extra10")
print("  corr(logD primary, logD dropout_off)=%.3f ; (primary,extra10)=%.3f ; (dropout_off,extra10)=%.3f"%tuple(np.corrcoef(np.log(a),np.log(b))[0,1] for a,b in ((Dp,Do),(Dp,Dx),(Do,Dx))))
print("  corr(logE primary, logE dropout_off)=%.3f"%np.corrcoef(np.log(Ep),np.log(Eo))[0,1])
print("  mean |log Dp - log Do| = %.3f"%np.mean(np.abs(np.log(Dp)-np.log(Do))))
out={}
print("\nM1-vs-M0 under different D/E sources (rel %, dMSE, CI, coef):")
for lab,(ds,es) in {"primary D & E (reported)":("primary","primary"),"dropout_off D & E":("dropout_off","dropout_off"),"extra10 D & E (10 samples)":("extra10","extra10"),
        "D primary -> E dropout_off (independent)":("primary","dropout_off"),"D dropout_off -> E primary (independent)":("dropout_off","primary"),
        "D primary -> E extra10":("primary","extra10"),"D dropout_off+extra10 -> E primary":(None,None),"pooled 50 D & E":("all","all"),"D pooled-excl -> E independent":(None,None)}.items():
    if ds is None: continue
    D,E=table(ds,es); s,ci,b=gain(D,E); out[lab]=dict(rel=s["rel"],dMSE=s["dMSE"],ci=list(ci),coef=b,rmse0=s["rmse0"])
    print(f"  {lab:42s} rel {s['rel']:5.2f}%  dMSE {s['dMSE']:7.4f} [{ci[0]:.4f},{ci[1]:.4f}]  coef {b:.3f}  M0 RMSE {s['rmse0']:.4f}")
# split-half within primary (shared coupling removed): D from samples 0-9, E from samples 10-19
D1=np.array([meanoff(M[t],np.arange(10)) for t in T]); E2=np.array([Ev[t][10:20].mean() for t in T]); s,ci,b=gain(D1,E2); print(f"  {'split-half primary (D 0-9 -> E 10-19)':42s} rel {s['rel']:5.2f}%  dMSE {s['dMSE']:7.4f} [{ci[0]:.4f},{ci[1]:.4f}]  coef {b:.3f}"); out["splithalf"]=dict(rel=s["rel"],dMSE=s["dMSE"],ci=list(ci),coef=b)
# 50-sample D with E from same - the coupling E>=D/sqrt2
Dall,Eall=table("all","all"); print("\nE vs D/√2: min E/(D/√2) = %.3f, fraction of targets with E < D/√2: %.1f%%, median E/D %.2f"%((Eall/(Dall/np.sqrt(2))).min(),100*np.mean(Eall<Dall/np.sqrt(2)),np.median(Eall/Dall)))
# medoid & best-of-N
med=[];best=[];mean=[]
for t in T:
    Mm=M[t][:20,:20]; i=np.argmin(Mm.sum(1)); med.append(Ev[t][:20][i]); best.append(Ev[t][:20].min()); mean.append(Ev[t][:20].mean())
    
med,best,mean=map(np.array,(med,best,mean)); print("medoid error (primary 20): mean %.2f vs ensemble-mean %.2f ; best-of-20 %.2f ; medoid better than mean in %.0f%% of targets"%(med.mean(),mean.mean(),best.mean(),100*np.mean(med<mean)))
print("medoid log-error LOOCV with D as predictor? (medoid error = outcome):")
for lab,Eo_ in (("medoid",med),("best-of-20",best)):
    s,ci,b=gain(table("primary","primary")[0],Eo_); print(f"  outcome {lab:11s}: rel {s['rel']:5.2f}% dMSE {s['dMSE']:.4f} [{ci[0]:.4f},{ci[1]:.4f}] coef {b:.3f}")
    out["outcome_"+lab]=dict(rel=s["rel"],ci=list(ci),coef=b)

# A9 ensemble size
print("\nA9 ensemble size (E from all 50 held fixed; D from n random samples, 40 draws)")
rng=np.random.default_rng(0); rows=[]
for n in (5,10,20,30,40):
    rel=[];coef=[];rho=[];cvD=[]
    for rep in range(40):
        Dn=np.array([meanoff(M[t],rng.choice(50,n,replace=False)) for t in T])
        s,ci,b=gain(Dn,Eall); rel.append(s["rel"]); coef.append(b); rho.append(spearmanr(Dn,Dall).correlation); 
    rows.append(dict(n=n,rel=np.mean(rel),rel_sd=np.std(rel),coef=np.mean(coef),rho_vs_full=np.mean(rho)))
    print(f"  n={n:2d}: rel {np.mean(rel):5.2f}% (SD {np.std(rel):.2f})  coef {np.mean(coef):.3f}  Spearman(D_n, D_50) {np.mean(rho):.3f}")
s,ci,b=gain(Dall,Eall); print(f"  n=50 (full): rel {s['rel']:.2f}%  coef {b:.3f}"); out["A9"]=rows
# FPC / D sampling CV: per-target sampling CV of D via subsampling n=20 from 50 (finite pool) -> compare with pairwise bootstrap
cv20=[]
for t in T:
    v=[meanoff(M[t],rng.choice(50,20,replace=False)) for _ in range(60)]; cv20.append(np.std(v)/np.mean(v))
print("  median per-target CV of D at n=20 from the 50-sample pool (includes FPC): %.1f%%  (mean %.1f%%)"%(100*np.median(cv20),100*np.mean(cv20)))
json.dump(out,open("a8_a9.json","w"),default=float,indent=1)
