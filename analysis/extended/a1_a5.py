import sys,json; sys.path.insert(0,".")
from lib import *
import statsmodels.api as sm
d=load(); y=d.logE.values; n=len(d); out={}
to=pd.DataFrame(json.load(open(SM+"training_overlap_detailed.json")))
G=set(to[to.excluded_at_80pct_threshold].target)
Lc=set(to[(~to.excluded_at_80pct_threshold)&(to.local_percent_identity>=.93)&(to.local_coverage_of_target>=.5)].target)
print("global",len(G),"local",len(Lc),sorted(Lc))

# ---------- A1 model table
X0,X1=design(d,"M0"),design(d,"M1")
res={}
for name,kind in [("length linear","L0"),("length+disp linear","L1"),("dispersion only","D"),("Model 0 spline(L)","M0"),("Model 1 spline(L)+logD","M1")]:
    X=design(d,kind); e=loo_fixed(X,y); res[name]=dict(loocv_rmse=float(np.sqrt(np.mean(e**2))))
    r=y-X@ols(X,y); res[name]["R2_in"]=float(1-r.var()/y.var())
out["A1_models"]=res
for k,v in res.items(): print(f"{k:28s} LOOCV RMSE {v['loocv_rmse']:.4f}  in-sample R2 {v['R2_in']:.3f}")
print("corr(logL,logD)=%.3f  spearman=%.3f"%(np.corrcoef(d.logL,d.logD)[0,1], d[["logL","logD"]].corr("spearman").iloc[0,1]))
m1=sm.OLS(y,X1).fit(); m0=sm.OLS(y,X0).fit()
b=m1.params[-1]
for lab,kw in [("classical",{}),("HC3",dict(cov_type="HC3")),("cluster",dict(cov_type="cluster",cov_kwds=dict(groups=d.cl.values)))]:
    mm=sm.OLS(y,X1).fit(**kw); ci=mm.conf_int()[-1]
    print(f"coef logD {mm.params[-1]:.3f}  SE({lab}) {mm.bse[-1]:.3f}  95%CI [{ci[0]:.3f},{ci[1]:.3f}] p={mm.pvalues[-1]:.3f}")
r_y=m0.resid; Xd=np.column_stack([X0]); r_d=d.logD.values-Xd@ols(Xd,d.logD.values)
pc=np.corrcoef(r_y,r_d)[0,1]; pr2=(m1.rsquared-m0.rsquared)/(1-m0.rsquared)
print(f"partial corr(logE,logD|spline L)={pc:.3f}  partial R2={pr2:.4f}  in-sample R2 M0 {m0.rsquared:.3f} M1 {m1.rsquared:.3f}")
out["A1_partial"]=dict(coef=float(b),partial_corr=float(pc),partial_R2=float(pr2),corrLD=float(np.corrcoef(d.logL,d.logD)[0,1]))
# leverage / influence on the logD coefficient
inf=m1.get_influence(); lev=inf.hat_matrix_diag; cook=inf.cooks_distance[0]
dfb=inf.dfbetas[:,-1]; dfbeta_raw=inf.dfbeta[:,-1]
dd=d.assign(lev=lev,cook=cook,dfbeta=dfbeta_raw).sort_values("dfbeta")
print("max leverage",lev.max().round(3),"(threshold 2p/n=%.3f)"%(2*X1.shape[1]/n)); print("most NEGATIVE dfbeta (removal raises coef):"); print(dd[["target","length","lev","cook","dfbeta"]].head(4).round(4).to_string(index=False))
print("most POSITIVE dfbeta (removal lowers coef):"); print(dd[["target","length","lev","cook","dfbeta"]].tail(4).round(4).to_string(index=False))
# symmetric influence: remove top-k each side by dfbeta, redo LOOCV delta (fixed knots recomputed)
def delta(sub):
    s=sub.reset_index(drop=True); yy=s.logE.values
    return summ(loo_fixed(design(s,"M0"),yy),loo_fixed(design(s,"M1"),yy))
sym={}
for k in (1,2,3,5):
    low=dd.target.head(k).tolist(); high=dd.target.tail(k).tolist()
    sym[k]=dict(remove_most_neg=delta(d[~d.target.isin(low)]),remove_most_pos=delta(d[~d.target.isin(high)]),remove_both=delta(d[~d.target.isin(low+high)]))
    print(f"k={k}: drop {k} most-negative-dfbeta -> rel {sym[k]['remove_most_neg']['rel']:.2f}% ; drop {k} most-positive -> {sym[k]['remove_most_pos']['rel']:.2f}% ; drop both -> {sym[k]['remove_both']['rel']:.2f}%")
# 9de8_A alone, and by LOOCV pointwise
e0=loo_fixed(X0,y); e1=loo_fixed(X1,y); dv=e0**2-e1**2
d["pdiff"]=dv
print("pointwise paired diffs, top 5:"); print(d.sort_values("pdiff",ascending=False)[["target","length","pdiff"]].head(5).round(4).to_string(index=False))
print("share of total gain from 9de8_A: %.1f%%"%(100*d.loc[d.target=='9de8_A','pdiff'].sum()/dv.sum() if (d.target=='9de8_A').any() else float('nan')))
print("without 9de8_A:",delta(d[d.target!='9de8_A']))
out["A1_sym"]={str(k):v for k,v in sym.items()}
d[["target","length","pdiff"]].to_json("pointwise.json",orient="records")

# ---------- A3 df table: fixed-knot LOOCV, fold-safe LOOCV, leave-cluster-out
print("\nA3 df table")
rows=[]
for df in (3,4,5,6,7):
    a,b_=loo_fixed(design(d,"M0",df),y),loo_fixed(design(d,"M1",df),y); s=summ(a,b_)
    ra,rb=cv_refit(d,"M0",df),cv_refit(d,"M1",df); s2=summ(ra,rb)
    ga,gb=cv_refit(d,"M0",df,groups=d.cl.values),cv_refit(d,"M1",df,groups=d.cl.values); s3=summ(ga,gb)
    ci=boot_ci(a**2-b_**2)
    rows.append(dict(df=df,rmse0=s["rmse0"],rmse1=s["rmse1"],rel=s["rel"],dMSE=s["dMSE"],ci_lo=ci[0],ci_hi=ci[1],rel_refit=s2["rel"],rmse0_refit=s2["rmse0"],rel_lco=s3["rel"],rmse0_lco=s3["rmse0"]))
t=pd.DataFrame(rows); print(t.round(4).to_string(index=False)); out["A3_df"]=rows
# varying coefficient: logD*(logL-c). compare against M0 AND M1
vc=loo_fixed(design(d,"Mvc"),y)
print("VC vs M0:",summ(e0,vc)); print("VC vs M1:",summ(e1,vc))
vc_ci0=boot_ci(e0**2-vc**2); vc_ci1=boot_ci(e1**2-vc**2); print("CI vs M0",vc_ci0,"vs M1",vc_ci1)
out["A3_vc"]=dict(vs_M0=summ(e0,vc),vs_M1=summ(e1,vc),ci_M0=list(vc_ci0),ci_M1=list(vc_ci1))
mv=sm.OLS(y,design(d,"Mvc")).fit(cov_type="HC3"); print("VC coefs logD, logD*(logL-c):",mv.params[-2:].round(3),"SE",mv.bse[-2:].round(3))
Lg=np.linspace(d.logL.min(),d.logL.max(),5); c=d.logL.mean()
V=mv.cov_params()[-2:,-2:]
print("slope of logD at length:")
for l in Lg:
    s=mv.params[-2]+mv.params[-1]*(l-c); v=np.array([1,l-c]); se=np.sqrt(v@V@v); print(f"  L={np.exp(l):6.0f}  beta={s:.3f}  [{s-1.96*se:.3f},{s+1.96*se:.3f}]")
out["A3_vc_curve"]=[dict(L=float(np.exp(l)),beta=float(mv.params[-2]+mv.params[-1]*(l-c))) for l in np.linspace(d.logL.min(),d.logL.max(),30)]
# fold-safe VC vs M0
vr=cv_refit(d,"Mvc"); r0=cv_refit(d,"M0"); print("VC fold-safe vs M0",summ(r0,vr))

# ---------- A4 Table 4 reconciliation
print("\nA4 Table 4")
t4=[]
for lab,excl in [("N=82",set()),("excl 15 global",G),("excl 21 global+local",G|Lc),("excl 6 local only",Lc)]:
    s=d[~d.target.isin(excl)].reset_index(drop=True); yy=s.logE.values
    a,b_=loo_fixed(design(s,"M0"),yy),loo_fixed(design(s,"M1"),yy); r=summ(a,b_)
    ci=boot_ci(a**2-b_**2)
    # also error in original Å for context
    t4.append(dict(cohort=lab,N=len(s),rmse0=r["rmse0"],rmse1=r["rmse1"],mse0=r["rmse0"]**2,mse1=r["rmse1"]**2,rel=r["rel"],dMSE=r["dMSE"],lo=ci[0],hi=ci[1],has9dib=bool((s.target=="9dib_A").any()),sumSE0=float((a**2).sum())))
print(pd.DataFrame(t4).round(4).to_string(index=False)); out["A4"]=t4
# the 6 removed targets' lengths and their place in the length range
print("removed-local lengths:",d[d.target.isin(Lc)][["target","length","error"]].round(1).to_string(index=False))
print("lengths of removed 21 vs kept 61: median", d[d.target.isin(G|Lc)].length.median(), d[~d.target.isin(G|Lc)].length.median())
# does the 61-subset gain rest on few targets?
s=d[~d.target.isin(G|Lc)].reset_index(drop=True); yy=s.logE.values
a,b_=loo_fixed(design(s,"M0"),yy),loo_fixed(design(s,"M1"),yy); s["p"]=a**2-b_**2
print("61-subset top pointwise gains:"); print(s.sort_values("p",ascending=False)[["target","length","p"]].head(4).round(3).to_string(index=False)); print("sum p",s.p.sum().round(3),"without top1 ->",(s.p.sum()-s.p.max()).round(3))
json.dump(out,open("a1_a5.json","w"),default=float,indent=1)
