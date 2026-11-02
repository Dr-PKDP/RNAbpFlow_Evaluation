import sys,json,time; sys.path.insert(0,".")
from lib import *
d=load(); y=d.logE.values; n=len(d); out={}
rng0=np.random.default_rng(7)
# ---------- repeated 5-fold CV: two implementations, 30 repeats
def kfold_folds(rng,n,K):
    p=rng.permutation(n); return [np.sort(p[i::K]) for i in range(K)]
def rep_cv(impl,R=30,K=5,seed=11,dd=d,df=4):
    rng=np.random.default_rng(seed); yy=dd.logE.values; rels=[]
    Xf0,Xf1=design(dd,"M0",df),design(dd,"M1",df)
    for r in range(R):
        folds=kfold_folds(rng,len(dd),K)
        if impl=="fixed":
            e0=np.zeros(len(dd)); e1=np.zeros(len(dd))
            for te in folds:
                tr=np.setdiff1d(np.arange(len(dd)),te)
                e0[te]=yy[te]-Xf0[te]@ols(Xf0[tr],yy[tr]); e1[te]=yy[te]-Xf1[te]@ols(Xf1[tr],yy[tr])
        else:
            e0=cv_refit(dd,"M0",df,folds=folds); e1=cv_refit(dd,"M1",df,folds=folds)
        rels.append(100*(1-np.sqrt(np.mean(e1**2)/np.mean(e0**2))))
    return np.array(rels)
print("Repeated 5-fold CV, 30 repeats")
for impl in ("fixed","refit"):
    for seed in (11,23,37):
        r=rep_cv(impl,seed=seed); print(f"  {impl:5s} seed {seed}: mean {r.mean():5.2f}%  SD {r.std(ddof=1):4.2f}%  range [{r.min():.1f},{r.max():.1f}]  worse in {100*np.mean(r<0):.1f}%")
out["rep5"]={impl:[rep_cv(impl,seed=s).tolist() for s in (11,23,37)] for impl in ("fixed","refit")}
r100=rep_cv("fixed",R=200,seed=5); print("  fixed 200 repeats: mean %.2f SD %.2f worse %.1f%%"%(r100.mean(),r100.std(ddof=1),100*np.mean(r100<0)))
r100b=rep_cv("refit",R=200,seed=5); print("  refit 200 repeats: mean %.2f SD %.2f worse %.1f%%"%(r100b.mean(),r100b.std(ddof=1),100*np.mean(r100b<0)))

# ---------- A5 GC on the same 81
gc=pd.DataFrame(json.load(open(SM+"dispersion_error_results_with_gc_renamed.json")))[["target","complexity"]]
d81=d.merge(gc,on="target").reset_index(drop=True); y81=d81.logE.values
print("\nA5: N =",len(d81),"; corr(complexity,logL)=%.2f"%np.corrcoef(d81.complexity,d81.logL)[0,1])
def X(dd,kind,df=4):
    S=design(dd,"M0",df); g=dd.complexity.values[:,None]; D=dd.logD.values[:,None]
    return {"M0":S,"M1":np.column_stack([S,D]),"MG":np.column_stack([S,g]),"MGD":np.column_stack([S,g,D])}[kind]
E={k:loo_fixed(X(d81,k),y81) for k in ("M0","M1","MG","MGD")}
for k,e in E.items(): print(f"  {k:3s} LOOCV RMSE {np.sqrt(np.mean(e**2)):.4f}")
print("  M0 vs M1 (81):",{k:round(float(v),4) for k,v in summ(E['M0'],E['M1']).items()}, "CI",boot_ci(E['M0']**2-E['M1']**2).round(4))
print("  MG vs MGD (81):",{k:round(float(v),4) for k,v in summ(E['MG'],E['MGD']).items()}, "CI",boot_ci(E['MG']**2-E['MGD']**2).round(4))
print("  M0 vs MG (81) [GC alone]:",{k:round(float(v),4) for k,v in summ(E['M0'],E['MG']).items()})
print("  M0 vs MGD:",{k:round(float(v),4) for k,v in summ(E['M0'],E['MGD']).items()})
out["A5"]={k:float(np.sqrt(np.mean(e**2))) for k,e in E.items()}

# ---------- A2 nested CV (Bates, Hastie & Tibshirani) for the paired loss difference
def fit_pred(dtr,dte,kind,df=4):
    Xtr=design(dtr,kind,df,fit=dtr); Xte=design(dte,kind,df,fit=dtr); return Xte@ols(Xtr,dtr.logE.values)
def dloss(dtr,dte,df=4):
    yte=dte.logE.values
    return (yte-fit_pred(dtr,dte,"M0",df))**2-(yte-fit_pred(dtr,dte,"M1",df))**2
def ncv(dd,K=10,R=60,seed=3,df=4):
    rng=np.random.default_rng(seed); nn=len(dd); a=[];b=[];cvmean=[];ncvmean=[];sec=[]
    for r in range(R):
        folds=kfold_folds(rng,nn,K); d_out_all=np.zeros(nn)
        for k in range(K):
            te=folds[k]; tr=np.concatenate([folds[j] for j in range(K) if j!=k])
            dtr,dte=dd.iloc[tr],dd.iloc[te]
            d_out=dloss(dtr,dte,df); d_out_all[te]=d_out
            d_in=[]
            for j in range(K):
                if j==k: continue
                inn_te=folds[j]; inn_tr=np.concatenate([folds[m] for m in range(K) if m not in (j,k)])
                d_in.append(dloss(dd.iloc[inn_tr],dd.iloc[inn_te],df))
            d_in=np.concatenate(d_in)
            a.append((d_in.mean()-d_out.mean())**2); b.append(d_out.var(ddof=1)/len(te)); ncvmean.append(d_in.mean())
        cvmean.append(d_out_all.mean()); sec.append(d_out_all.std(ddof=1)/np.sqrt(nn))
    mse_ncv=np.mean(a)-np.mean(b); mse=(K-1)/K*mse_ncv
    secv=float(np.mean(sec)); se=float(np.sqrt(max(mse,0))); se_c=min(max(se,secv),np.sqrt(K)*secv)
    return dict(cv=float(np.mean(cvmean)),ncv_point=float(np.mean(ncvmean)),naive_se=secv,ncv_se_raw=se,ncv_se=se_c,inflation=se_c/secv,mse_raw=float(mse))
t=time.time(); res=ncv(d); print("\nNested CV (K=10, R=60) on 82:",{k:round(v,5) for k,v in res.items()},"%.0fs"%(time.time()-t))
cv,se=res["cv"],res["ncv_se"]; print("  95%% interval for expected dMSE: [%.4f, %.4f]"%(cv-1.96*se,cv+1.96*se))
base=0.3609**2
print("  as relative RMSE %%: approx [%.1f, %.1f]"%(100*(1-np.sqrt(1-(cv+1.96*se)/base)) *-1 if False else 100*(1-np.sqrt(1-(cv-1.96*se)/base)), 100*(1-np.sqrt(1-(cv+1.96*se)/base))))
out["A2_ncv"]=res
json.dump(out,open("a2_a5.json","w"),default=float,indent=1)
