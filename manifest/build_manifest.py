import os,sys,json,csv
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,'..','analysis','extended'))
from lib import *
S=os.environ.get('RNA_DATA','data').rstrip('/')+'/'
d=load(); y=d.logE.values
meta=json.load(open(S+'bundle/meta.json'))
info={l.split('\t')[0]:l.rstrip('\n').split('\t') for l in open(S+'rcsb_info.tsv')}
ov={r['target']:r for r in json.load(open(S+'small/training_overlap_detailed.json'))}
c2={r['target']:r for r in json.load(open(S+'c2/c2_real_flags.json')) if r['kind']=='real' and r['rep']==0}
f=json.load(open(S+'rf/rfam_scan/families.json'))
tr={k:v for k,v in f.items() if k[:2] not in('T:','X:')}; trfam={v['top']['acc'] for v in tr.values() if v.get('top')}
cl=json.load(open(S+'small/grouped_cv_results.json'))['clusters']
# deterministic cluster numbering: order of first appearance in the cohort order
order={}
for t in d.target:
    for i,c in enumerate(cl):
        if t in c: order.setdefault(i,len(order)+1)
cid={t:order[i] for i,c in enumerate(cl) for t in c}
e0=loo_fixed(design(d,'M0'),y); e1=loo_fixed(design(d,'M1'),y)
p0=np.exp(y-e0); p1=np.exp(y-e1)
rows=[]
for i,r in d.iterrows():
    t=r.target; m=meta[t]; pid=m['pdb_id']; rel,meth,res=info[pid][1:4]
    o=ov.get(t); c=c2[t]
    fam=f['T:'+t]; fs='no Rfam hit' if not fam.get('top') else ('family in training set' if fam['top']['acc'] in trfam else 'family absent from training set')
    flags=''.join([('A' if c['flag80'] else ''),('F' if c['flagloc'] else ''),('N' if c['cal99'] else ''),('R' if fs=='family in training set' else '')])
    rows.append(dict(target=t,pdb_id=pid,chain=m['chain'],release_date=rel,method=meth,resolution_A=res,
      entity_length_nt=m['n_entity'],resolved_residues=m['n_paired'],resolved_fraction=round(m['n_paired']/m['n_entity'],3),
      hetatm_c4_residues=m['n_het_c4'],sequence_cluster_80pct=cid[t],
      dispersion_D_A=round(r.dispersion,4),error_E_A=round(r.error,4),
      oof_pred_E_M0_A=round(float(p0[i]),4),oof_pred_E_M1_A=round(float(p1[i]),4),
      oof_logE_M0=round(float(y[i]-e0[i]),5),oof_logE_M1=round(float(y[i]-e1[i]),5),
      pointwise_gain_se0_minus_se1=round(float(e0[i]**2-e1[i]**2),5),
      nearest_training_chain=c['nearest'],global_identity=round(c['global_id'],4),local_identity=round(c['local_id'],4),
      local_coverage_of_target=round(c['local_cov_target'],4),
      flag_global_ge80=bool(c['flag80']),flag_local_fragment=bool(c['flagloc']),flag_above_shuffle_null_p99=bool(c['cal99']),
      rfam_status=fam and fs,flag_letters=flags))
cols=list(rows[0].keys())
with open(os.path.join(HERE,'manifest_82_targets.csv'),'w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=cols); w.writeheader(); w.writerows(rows)
json.dump(rows,open(S+'rows.json','w'))
print(len(rows), 'flags A',sum(r['flag_global_ge80'] for r in rows),'F',sum(r['flag_local_fragment'] for r in rows),'N',sum(r['flag_above_shuffle_null_p99'] for r in rows))
import collections; print(collections.Counter(r['rfam_status'] for r in rows))
print(collections.Counter(r['method'] for r in rows), min(r['release_date'] for r in rows), max(r['release_date'] for r in rows))
print(rows[0])
# candidate-level
v2=json.load(open(S+'small/dispersion_error_results.json')); keptset=set(d.target)
c3=json.load(open(S+'c13/t/c3_facts.json'))
cand=[]
chain151={r['target'].split('_')[0].upper():r for r in v2}
c45={r['pdb_id']:r for r in c3}
for pid,v in info.items():
    rel,meth,res=v[1:4]
    if pid in chain151:
        r=chain151[pid]; t=r['target']
        if t in keptset: st='included in cohort (82)'
        else: st='excluded: exact-sequence duplicate of an earlier chain (69)'
        cand.append(dict(pdb_id=pid,chain=t.split('_')[1],release_date=rel,method=meth,resolution_A=res,status=st,resolved_residues=r['length'],D_A=round(float(r['dispersion']),4),E_A=round(float(r['error']),4)))
    else:
        cand.append(dict(pdb_id=pid,chain=(c45[pid]['chain'] if pid in c45 else ''),release_date=rel,method=meth,resolution_A=res,status='excluded: mmCIF-only, not processed by the coordinate pipeline (45)',resolved_residues=(c45[pid]['length'] if pid in c45 else ''),D_A='',E_A=''))
dup=json.load(open(S+'small/duplicate_of.json'))
for c in cand: c['duplicate_of']=dup.get('%s_%s'%(c['pdb_id'].lower(),c['chain']),'')
cand.sort(key=lambda r:r['pdb_id'])
with open(os.path.join(HERE,'candidates_196_entries.csv'),'w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(cand[0].keys())); w.writeheader(); w.writerows(cand)
print(collections.Counter(c['status'] for c in cand), len(cand))
