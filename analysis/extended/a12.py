import sys,json,time; sys.path.insert(0,".")
from lib import *; from bund import META,gt; from metrics import tm_score
from Bio import Align
import scipy.sparse as sp; from scipy.sparse.csgraph import connected_components
d=load(); T=d.target.tolist(); n=len(T)
al=Align.PairwiseAligner(); al.mode="global"; al.match_score=2; al.mismatch_score=-1; al.open_gap_score=-2; al.extend_gap_score=-0.5
S={t:META[t]["gt_seq"] for t in T}; G={t:gt(t) for t in T}
t0=time.time(); rows=[]
for i in range(n):
    for j in range(i+1,n):
        a,b=T[i],T[j]; sa,sb=S[a],S[b]
        if min(len(sa),len(sb))<8: continue
        # cheap pre-filter: skip when lengths differ >2x
        if max(len(sa),len(sb))>2.0*min(len(sa),len(sb)): continue
        aln=al.align(sa,sb)[0]; A,B=aln[0],aln[1]; pa=pb=0; ia=[];ib=[];mm=0
        for x,y in zip(A,B):
            if x!="-" and y!="-":
                ia.append(pa); ib.append(pb); mm+= (x!=y)
            pa+= x!="-"; pb+= y!="-"
        k=len(ia)
        if k<0.7*min(len(sa),len(sb)) or mm/k>0.5: continue
        tm=tm_score(G[a][ia],G[b][ib])   # normalised by aligned length
        # normalise by the shorter chain length instead
        tmn=tm*k/min(len(sa),len(sb))
        rows.append(dict(a=a,b=b,k=k,ident=1-mm/k,tm_aligned=tm,tm_short=tmn))
P=pd.DataFrame(rows); print("pairs evaluated %d in %.0fs"%(len(P),time.time()-t0))
P.to_json("a12_pairs.json",orient="records")
print(P[["ident","tm_aligned","tm_short"]].describe().round(2).loc[["count","mean","50%","max"]])
cl_seq=d.cl.values
def clusters(thr,col="tm_short"):
    e=P[P[col]>=thr]; idx={t:i for i,t in enumerate(T)}
    M=sp.coo_matrix((np.ones(len(e)),([idx[x] for x in e.a],[idx[x] for x in e.b])),shape=(n,n))
    return connected_components(M,directed=False)[1]
print("sequence clusters (80%%): %d (multi-member 13)"%len(np.unique(cl_seq)))
y=d.logE.values
def lco_fixed(groups,kind):
    X=design(d,kind); e=np.zeros(n)
    for c in np.unique(groups):
        te=np.where(groups==c)[0]; tr=np.setdiff1d(np.arange(n),te); e[te]=y[te]-X[te]@ols(X[tr],y[tr])
    return e
for thr in (0.5,0.4,0.3):
    g=clusters(thr); sizes=np.bincount(g); a=lco_fixed(g,"M0"); b=lco_fixed(g,"M1"); s=summ(a,b); ci=boot_ci(a**2-b**2,g)
    print(f"  structural clusters TM>={thr} (sequence-guided): {len(np.unique(g))} clusters, largest {sizes.max()}; LCO rel {s['rel']:.2f}% dMSE {s['dMSE']:.4f} cluster-CI [{ci[0]:.4f},{ci[1]:.4f}]")
    if thr==0.5: d["scl"]=g
# do structural clusters merge sequence clusters (cross-cluster links)?
e=P[P.tm_short>=0.5]; idx=dict(zip(T,cl_seq)); cross=e[[idx[a]!=idx[b] for a,b in zip(e.a,e.b)]]
print("pairs with TM>=0.5 across different 80%-sequence clusters:",len(cross)); print(cross.sort_values("tm_short",ascending=False).head(8).round(2).to_string(index=False))
