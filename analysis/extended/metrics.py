import numpy as np
def d0_rna(L):
    if L<12: return 0.3
    if L<16: return 0.4
    if L<20: return 0.5
    if L<24: return 0.6
    if L<30: return 0.7
    return 0.6*np.sqrt(L-0.5)-2.5
def kabsch(P,Q,w=None):
    """rotation/translation superposing P onto Q (rows)"""
    pc,qc=P.mean(0),Q.mean(0); H=(P-pc).T@(Q-qc); U,S,Vt=np.linalg.svd(H)
    d=np.sign(np.linalg.det(Vt.T@U.T)); D=np.diag([1,1,d]); R=Vt.T@D@U.T
    return R,qc-R@pc
def tm_score(X,Y):
    """X mobile, Y reference, residue-paired (n,3). TM normalised by n, RNA d0."""
    n=len(X); d0=d0_rna(n); best=0
    dsrch=min(max(d0+0.0,4.5),8.0) if n>=30 else 4.5
    Ls=sorted({n,max(n//2,4),max(n//4,4),max(n//8,4),4})
    for Lf in Ls:
        if Lf<3: continue
        step=max(1,Lf//2) if Lf<n else 1
        for s in range(0,n-Lf+1,step):
            idx=np.arange(s,s+Lf)
            for it in range(20):
                if len(idx)<3: break
                R,t=kabsch(X[idx],Y[idx]); Xt=X@R.T+t; dd=np.linalg.norm(Xt-Y,axis=1)
                sc=np.mean(1/(1+(dd/d0)**2)); best=max(best,sc)
                dc=dsrch
                while True:
                    new=np.where(dd<dc)[0]
                    if len(new)>=3 or dc>dsrch+20: break
                    dc+=0.5
                if len(new)<3 or (len(new)==len(idx) and np.all(new==idx)): break
                idx=new
            if n>600 and Lf<n: break
    return best
def lddt(X,Y,R0=15.0,th=(0.5,1,2,4)):
    """C4' lDDT of model X vs reference Y, inclusion radius R0, excludes |i-j|<1"""
    dY=np.linalg.norm(Y[:,None]-Y[None],axis=-1); dX=np.linalg.norm(X[:,None]-X[None],axis=-1)
    m=(dY<R0)&(~np.eye(len(Y),dtype=bool)); diff=np.abs(dX-dY)
    sc=np.mean([ (diff<t) for t in th],axis=0)
    per=np.array([ (sc[i][m[i]].mean() if m[i].any() else np.nan) for i in range(len(Y))])
    return np.nanmean(per)
def contact_f1(X,Y,cut=12.0,minsep=4,idx=None):
    """Contact F1 over C4' pairs closer than `cut` A. `idx` holds the positions of the paired
    residues in the deposited sequence; sequence separation is applied to these positions, so
    unresolved residues do not shorten the separation. Without idx the array order is used."""
    n=len(Y)
    if idx is None:
        i,j=np.triu_indices(n,minsep)
    else:
        i,j=np.triu_indices(n,1); keep=(np.asarray(idx)[j]-np.asarray(idx)[i])>=minsep; i,j=i[keep],j[keep]
    dY=np.linalg.norm(Y[i]-Y[j],axis=1)<cut; dX=np.linalg.norm(X[i]-X[j],axis=1)<cut
    tp=(dY&dX).sum(); p=tp/max(dX.sum(),1); r=tp/max(dY.sum(),1)
    return 2*p*r/(p+r) if p+r>0 else 0.0
