import sys,json; sys.path.insert(0,".")
from lib import *
from varying_coefficient_model_foldsafe import loocv_varying_coefficient_foldsafe as vc
d=load(); out={}
for df in [3,4,5,6,7]:
    se1,sevc=vc(d.length.values,d.dispersion.values,d.error.values,main_spline_df=df)
    e0=cv_refit(d,'M0',df); se0=e0**2
    r=dict(rmse0=float(np.sqrt(se0.mean())),rmse1=float(np.sqrt(se1.mean())),rmsevc=float(np.sqrt(sevc.mean())))
    for lab,a in [("vs0",se0),("vs1",se1)]:
        dv=a-sevc; ci=boot_ci(dv,B=10000); cc=boot_ci(dv,groups=d.cl.values,B=4000)
        top=np.sort(dv)[::-1]
        r[lab]=dict(dMSE=float(dv.mean()),ci=ci.tolist(),cl=cc.tolist(),rel=float(100*(1-np.sqrt(sevc.mean())/np.sqrt(a.mean()))),top3=float(top[:3].sum()/dv.sum()),rest=float(dv.sum()-top[:3].sum()))
    out[df]=r
    print(df,json.dumps(r))
json.dump(out,open("vc_df.json","w"),indent=1)
