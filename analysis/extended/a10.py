import sys; sys.path.insert(0,".")
from lib import *; from bund import META, get, gt, Z
from scipy.stats import spearmanr
P=np.load("pairwise.npz"); d=load(); T=d.target.tolist(); y=d.logE.values
rows=[]
for t in T:
    m=META[t]; Lp=len(m["pred_seq"]); rows.append(dict(target=t,n_pred=Lp,n_paired=m["n_paired"],n_atom=m["n_atom_residues"],n_c4=m["n_c4_all_chain"],het=m["n_het_c4"],mis=m["mismatch_rate"],v1=m["v1_order_pairing_would_pass"],cov=m["n_paired"]/Lp))
a=pd.DataFrame(rows); d=d.merge(a,on="target")
print("n_paired==length column:",(d.n_paired==d.length).all())
print("mismatch rate: max %.3f, targets >0: %d"%(a.mis.max(),(a.mis>0).sum()))
print("coverage (resolved/predicted): median %.2f, min %.2f, <0.5: %d, ==1: %d"%(a["cov"].median(),a["cov"].min(),(a["cov"]<.5).sum(),(a["cov"]==1).sum()))
print("v1 accepted: %d; coverage==1: %d; v1 accepted & cov<1: %d"%(a.v1.sum(),(a["cov"]==1).sum(),((a.v1)&(a["cov"]<1)).sum()))
print("HETATM-C4' targets:",a.het.gt(0).sum(),"; their coverage:",a[a.het>0]["cov"].round(2).tolist())
print("n_c4 (all chain, incl HETATM) vs n_paired where het>0:", a[a.het>0][["target","n_c4","n_paired","het"]].values.tolist()[:6])
# D over all predicted residues vs resolved (primary run)
from bund import rmsd_pairs
Dall=[];Dres=[]
for t in T:
    X=get(t,"primary",paired=False)[:20]; Dall.append(rmsd_pairs(X,X)[np.triu_indices(20,1)].mean()); Dres.append(P[f"{t}__M"][:20,:20][np.triu_indices(20,1)].mean())
d["D_all"]=Dall; d["D_res"]=Dres
print("D(all predicted residues)/D(resolved): median %.2f, range %.2f–%.2f ; corr(log) %.3f"%(np.median(d.D_all/d.D_res),(d.D_all/d.D_res).min(),(d.D_all/d.D_res).max(),np.corrcoef(np.log(d.D_all),np.log(d.D_res))[0,1]))
# model with D over all residues; and with length = full predicted length
def gainX(dd):
    yy=dd.logE.values; e0=loo_fixed(design(dd,"M0"),yy); e1=loo_fixed(design(dd,"M1"),yy); s=summ(e0,e1); return s,boot_ci(e0**2-e1**2)
for lab,mod in (("primary (resolved L, resolved D)",d),
                ("resolved L, D over all predicted residues",d.assign(logD=np.log(d.D_all))),
                ("predicted-length L, resolved D",d.assign(logL=np.log(d.n_pred))),
                ("predicted-length L, D over all residues",d.assign(logL=np.log(d.n_pred),logD=np.log(d.D_all)))):
    s,ci=gainX(mod); print(f"  {lab:44s} M0 RMSE {s['rmse0']:.4f} M1 {s['rmse1']:.4f}  rel {s['rel']:5.2f}%  dMSE {s['dMSE']:.4f} [{ci[0]:.4f},{ci[1]:.4f}]")
# coverage as covariate
d["logcov"]=np.log(d["cov"])
S=design(d,"M0"); X=np.column_stack([S,d.logcov]); X2=np.column_stack([S,d.logcov,d.logD])
e0=loo_fixed(S,y); ec=loo_fixed(X,y); ecd=loo_fixed(X2,y)
print("  + coverage covariate: M0 %.4f, +cov %.4f, +cov+D %.4f ; D gain given cov rel %.2f%%  dMSE %.4f"%(np.sqrt((e0**2).mean()),np.sqrt((ec**2).mean()),np.sqrt((ecd**2).mean()),summ(ec,ecd)["rel"],summ(ec,ecd)["dMSE"]))
print("  corr(logcov, logD)=%.2f, corr(logcov,logE)=%.2f"%(np.corrcoef(d.logcov,d.logD)[0,1],np.corrcoef(d.logcov,d.logE)[0,1]))
# HETATM-bearing targets excluded
s,ci=gainX(d[d.het==0].reset_index(drop=True)); print("  excluding 12 HETATM-C4' targets (N=%d): rel %.2f%% dMSE %.4f [%.4f,%.4f]"%((d.het==0).sum(),s["rel"],s["dMSE"],*ci))
# targets where v1 would have accepted only
s,ci=gainX(d[d.v1].reset_index(drop=True)); print("  v1-acceptable targets only (N=%d): rel %.2f%%  [%.4f,%.4f]"%(d.v1.sum(),s["rel"],*ci))
s,ci=gainX(d[d["cov"]>=0.9].reset_index(drop=True)); print("  coverage>=0.9 (N=%d): rel %.2f%%  [%.4f,%.4f]"%((d["cov"]>=.9).sum(),s["rel"],*ci))
print("9dib_A:",d[d.target=="9dib_A"][["n_pred","n_paired","n_atom","n_c4","cov"]].values.tolist())
d.drop(columns=[]).to_json("a10_table.json",orient="records")
