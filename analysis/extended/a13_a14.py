import os,sys,json; sys.path.insert(0,".")
from lib import *; from bund import META, Z
BP=np.load(os.environ.get("RNA_DATA", "data").rstrip("/") + "/bundle/bp.npz")
d=load(); T=d.target.tolist()
def pairset(a): return {(int(min(r[0],r[1])),int(max(r[0],r[1]))) for r in a}
def f1(pred,ref):
    if not pred or not ref: return 0.0,0.0,0.0
    tp=len(pred&ref); p=tp/len(pred); r=tp/len(ref); return (2*p*r/(p+r) if p+r else 0.0),p,r
def helices(ps):
    s=sorted(ps); n=0; seen=set()
    for (i,j) in s:
        if (i,j) in seen: continue
        n+=1; a,b=i,j
        while (a+1,b-1) in ps: a,b=a+1,b-1; seen.add((a,b))
    return n
def pk(ps):
    s=sorted(ps)
    for x,(i,j) in enumerate(s):
        for (k,l) in s[x+1:]:
            if k>j: break
            if i<k<j<l: return 1
    return 0
rows=[]
for t in T:
    m=META[t]; res=set(map(int,Z[f"{t}__pred_idx"])); ref={p for p in pairset(BP[f"{t}__ref_pairs"]) }
    maps=[pairset(BP[f"{t}__map{k}"]) for k in (1,2,3)]
    cnt={}
    for mp in maps:
        for p in mp: cnt[p]=cnt.get(p,0)+1
    cons={p for p,c in cnt.items() if c>=2}; union=set(cnt)
    L=len(m["pred_seq"]); inres=lambda ps:{p for p in ps if p[0] in res and p[1] in res}
    r=dict(target=t)
    for k,mp in enumerate(maps,1): r[f"f1_m{k}"]=f1(inres(mp),ref)[0]
    r["f1_cons"]=f1(inres(cons),ref)[0]; r["f1_union"]=f1(inres(union),ref)[0]
    r["prec_cons"],r["rec_cons"]=f1(inres(cons),ref)[1:]
    j=[len(a&b)/max(len(a|b),1) for a,b in ((maps[0],maps[1]),(maps[0],maps[2]),(maps[1],maps[2]))]; r["agree"]=np.mean(j)
    r["paired_frac"]=2*len(cons)/L; r["helices"]=helices(cons); r["pk"]=pk(cons)
    r["longrange"]=np.mean([(b-a)>L/2 for a,b in cons]) if cons else 0.0; r["nref"]=len(ref)
    rows.append(r)
A=pd.DataFrame(rows); d=d.merge(A,on="target")
print("A14 input-map accuracy vs approximate reference (resolved residues only), mean/median F1:")
for k in ("f1_m1","f1_m2","f1_m3","f1_cons","f1_union"): print(f"  {k:9s} mean {d[k].mean():.3f} median {d[k].median():.3f}")
print("  mean precision/recall of consensus: %.3f / %.3f ; map agreement (Jaccard) %.3f"%(d.prec_cons.mean(),d.rec_cons.mean(),d.agree.mean()))
print("  targets with consensus F1<0.3:",(d.f1_cons<.3).sum(),"; pseudoknot indicator n=%d"%d.pk.sum())
from scipy.stats import spearmanr
for c in ("f1_cons","agree","paired_frac","helices","longrange"):
    print(f"  Spearman({c}, logE) {spearmanr(d[c],d.logE).correlation:6.3f} ; with logD {spearmanr(d[c],d.logD).correlation:6.3f} ; with logL {spearmanr(d[c],d.logL).correlation:6.3f}")
# A13 nested hierarchy (input-map-only covariates; no reference info)
y=d.logE.values; S=design(d,"M0")
covs={"paired_frac":d.paired_frac.values,"log_helices":np.log1p(d.helices.values),"longrange":d.longrange.values,"agree":d.agree.values}
C=np.column_stack(list(covs.values())); Cs=(C-C.mean(0))/C.std(0)
to=pd.DataFrame(json.load(open(SM+"training_overlap_detailed.json")))[["target","global_percent_identity"]]
def run(X,lab,base):
    e=loo_fixed(X,y); print(f"  {lab:46s} LOOCV RMSE {np.sqrt((e**2).mean()):.4f}  vs prior: rel {summ(base,e)['rel']:6.2f}%"); return e
print("\nA13 nested hierarchy (N=82):")
e0=run(S,"M0 spline(L)",loo_fixed(S,y))
eC=run(np.column_stack([S,Cs]),"+ 4 pre-inference map covariates",e0)
eCD=run(np.column_stack([S,Cs,d.logD.values]),"+ log D",eC)
s=summ(eC,eCD); print("   D given covariates: dMSE %.4f CI"%s["dMSE"],boot_ci(eC**2-eCD**2).round(4), "cluster",boot_ci(eC**2-eCD**2,d.cl.values).round(4))
for k,v in covs.items():
    z=(v-v.mean())/v.std(); e1=loo_fixed(np.column_stack([S,z]),y); print(f"   single: +{k:12s} RMSE {np.sqrt((e1**2).mean()):.4f}  rel {summ(e0,e1)['rel']:6.2f}%")
d81=d.merge(to,on="target").reset_index(drop=True); y81=d81.logE.values; S81=design(d81,"M0"); idn=((d81.global_percent_identity-d81.global_percent_identity.mean())/d81.global_percent_identity.std()).values
a=loo_fixed(S81,y81); b=loo_fixed(np.column_stack([S81,idn]),y81); c=loo_fixed(np.column_stack([S81,idn,d81.logD.values]),y81)
print("   nearest-training identity (N=81): M0 %.4f, +identity %.4f (rel %.2f%%), +identity+D %.4f (D given identity rel %.2f%%, dMSE %.4f)"%(np.sqrt((a**2).mean()),np.sqrt((b**2).mean()),summ(a,b)["rel"],np.sqrt((c**2).mean()),summ(b,c)["rel"],summ(b,c)["dMSE"]))
d.to_json("a13_table.json",orient="records")
