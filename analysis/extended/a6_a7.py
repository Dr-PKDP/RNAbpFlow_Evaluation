import sys,json; sys.path.insert(0,".")
from lib import *
d=load(); y=d.logE.values; n=len(d); E=d.error.values; out={}
# cross-fitted (LOOCV, fixed knots) predicted log-error
p0=y-loo_fixed(design(d,"M0"),y); p1=y-loo_fixed(design(d,"M1"),y)
pD=y-loo_fixed(design(d,"D"),y)   # dispersion-only model
def aurc(score,err,risk="mean"):
    o=np.argsort(score); e=err[o]; k=np.arange(1,len(e)+1)
    if risk=="mean": R=np.cumsum(e)/k
    elif risk=="median": R=np.array([np.median(e[:i]) for i in k])
    return R.mean()   # mean of R(c_k), c_k=k/n
pol={"length":p0,"dispersion":d.dispersion.values,"dispersion_model":pD,"combined":p1}
print("mean error (random policy AURC): %.2f ; oracle: %.2f"%(E.mean(),aurc(E,E)))
for k,s in pol.items(): print(f"  AURC {k:17s} mean-risk {aurc(s,E):.2f}  median-risk {aurc(s,E,'median'):.2f}")
# reported Table 5: 17.74/20.51/18.10
out["A6_point"]={k:aurc(s,E) for k,s in pol.items()}; out["A6_oracle"]=aurc(E,E); out["A6_random"]=float(E.mean())
# cluster bootstrap of differences
cl=d.cl.values; u=np.unique(cl); mem=[np.where(cl==k)[0] for k in u]; rng=np.random.default_rng(2); B=4000
diffs={"L-C":[],"L-D":[],"L-oracle gap":[]}
frac=[]
for b in range(B):
    idx=np.concatenate([mem[k] for k in rng.integers(0,len(u),len(u))])
    e=E[idx]; aL=aurc(p0[idx],e); aC=aurc(p1[idx],e); aD=aurc(pol["dispersion"][idx],e); aO=aurc(e,e)
    diffs["L-C"].append(aL-aC); diffs["L-D"].append(aL-aD); frac.append((aL-aO)/(e.mean()-aO))
for k in ("L-C","L-D"): print(f"  cluster-bootstrap Δ(length−{k[-1]}) mean {np.mean(diffs[k]):.2f}  95% CI [{np.percentile(diffs[k],2.5):.2f},{np.percentile(diffs[k],97.5):.2f}]")
print("  share of oracle gain achieved by length ranking: %.1f%%"%(100*(1-np.mean(frac))))
out["A6_boot"]={k:[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))] for k,v in diffs.items() if k!="L-oracle gap"}
# one-policy sanity: how much of the AURC difference is from the extreme-error target(s)
print("  top-3 errors (Å):",np.sort(E)[-3:].round(1), d.loc[np.argsort(E)[-3:],'target'].tolist())

# relative-RMSE interval, cluster bootstrap of LOOCV loss pairs (fixed knots)
e0=loo_fixed(design(d,"M0"),y); e1=loo_fixed(design(d,"M1"),y); rel=[]
for b in range(B):
    idx=np.concatenate([mem[k] for k in rng.integers(0,len(u),len(u))])
    rel.append(100*(1-np.sqrt(np.mean(e1[idx]**2)/np.mean(e0[idx]**2))))
print("\nrelative RMSE reduction, cluster bootstrap 95%% CI: [%.1f, %.1f]"%tuple(np.percentile(rel,[2.5,97.5])))
rel_t=[]
for b in range(B):
    idx=rng.integers(0,n,n); rel_t.append(100*(1-np.sqrt(np.mean(e1[idx]**2)/np.mean(e0[idx]**2))))
print("relative RMSE reduction, target bootstrap 95%% CI: [%.1f, %.1f]"%tuple(np.percentile(rel_t,[2.5,97.5])))
out["rel_ci"]=dict(cluster=list(np.percentile(rel,[2.5,97.5])),target=list(np.percentile(rel_t,[2.5,97.5])))

# ---------- A7 conformal, 3-way split (fit/cal/test), residual score on log scale
L=d.length.values; med=np.median(L); short=L<=med
def split_conformal(R=2000,nfit=34,ncal=24,alpha=.1,seed=4,norm=False):
    rng=np.random.default_rng(seed); rows=[]
    for r in range(R):
        p=rng.permutation(n); fit,cal,te=p[:nfit],p[nfit:nfit+ncal],p[nfit+ncal:]
        row={}
        for kind in ("M0","M1"):
            dtr=d.iloc[fit]; Xtr=design(dtr,kind,4,fit=dtr); b=ols(Xtr,y[fit])
            pc=design(d.iloc[cal],kind,4,fit=dtr)@b; pt=design(d.iloc[te],kind,4,fit=dtr)@b
            if norm:   # scale = dispersion relative to training median dispersion
                sc=lambda ix: d.dispersion.values[ix]/np.median(d.dispersion.values[fit])
                s=np.abs(y[cal]-pc)/sc(cal); k=int(np.ceil((ncal+1)*(1-alpha))); q=np.sort(s)[min(k,ncal)-1]
                lo=pt-q*sc(te); hi=pt+q*sc(te)
            else:
                s=np.abs(y[cal]-pc); k=int(np.ceil((ncal+1)*(1-alpha))); q=np.sort(s)[min(k,ncal)-1]
                lo=pt-q; hi=pt+q
            w=np.exp(hi)-np.exp(lo); cov=(y[te]>=lo)&(y[te]<=hi)
            row[kind]=dict(cov=cov.mean(),wmean=w.mean(),wmed=np.median(w),wshort=np.median(w[short[te]]) if short[te].any() else np.nan,wlong=np.median(w[~short[te]]) if (~short[te]).any() else np.nan,
                           corr=np.corrcoef(np.log(L[te]),np.log(w))[0,1])
        rows.append(row)
    return rows
for lab,nm in (("unnormalised",False),("normalised by D",True)):
    rows=split_conformal(norm=nm)
    f=lambda kind,key: np.nanmean([r[kind][key] for r in rows])
    dm=np.array([r["M0"]["wmean"]-r["M1"]["wmean"] for r in rows]); dmed=np.array([r["M0"]["wmed"]-r["M1"]["wmed"] for r in rows])
    print(f"\nConformal ({lab}), {len(rows)} splits, 24 calibration targets, nominal 90% (finite-sample guarantee ≥ 92.0%)")
    for kind in ("M0","M1"):
        print(f"  {kind}: coverage {100*f(kind,'cov'):.1f}%  mean width {f(kind,'wmean'):.1f} Å  median-of-median width {f(kind,'wmed'):.1f}  short-stratum median {f(kind,'wshort'):.1f}  long-stratum median {f(kind,'wlong'):.1f}  corr(log L,log width) {f(kind,'corr'):.2f}")
    print(f"  paired Δwidth(M0−M1): mean {dm.mean():.2f} Å, median {np.median(dm):.2f}; of medians: mean {dmed.mean():.2f}, median {np.median(dmed):.2f}; M1 narrower in {100*np.mean(dm>0):.1f}% of splits (mean-width), {100*np.mean(dmed>0):.1f}% (median-width)")
    out["A7_"+lab]={k:{kk:float(f(k,kk)) for kk in ("cov","wmean","wmed","wshort","wlong","corr")} for k in ("M0","M1")}
    out["A7_"+lab]["paired"]=dict(dmean=float(dm.mean()),dmed_mean=float(dmed.mean()),frac_narrower_mean=float(np.mean(dm>0)),frac_narrower_med=float(np.mean(dmed>0)))
json.dump(out,open("a6_a7.json","w"),default=float,indent=1)
