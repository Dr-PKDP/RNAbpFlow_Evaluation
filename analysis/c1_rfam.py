"""
c1_rfam.py -- Rfam family assignment (Infernal cmscan) for the evaluation targets, the extended-cohort
entries and the RNA3DB training chains, so that family-level overlap with the training set can be assessed.

Phases (all idempotent, resumable):
  prep     build query FASTA batches   (targets from the exported bundle meta.json, extended cohort from
                                         c3_facts.json, training chains from the RNA3DB chains.tsv, split=train_set;
                                         sequences longer than WINDOW nt are cut into overlapping windows)
  run      cmscan on every batch, in parallel (--procs), skipping batches whose .tbl already exists
  collect  parse the --tblout files into families.json  (best hit and all hits per original sequence)

Settings: cmscan --cut_ga --rfam --nohmmonly --noali (Rfam gathering thresholds, as recommended for Rfam;
RNA3DB's own run used cmscan with Rfam 14.10 and a second --max pass for sequences without a hit, so
assignments here are an approximation of the RNA3DB family labels, not a reproduction of them).

Usage (from backbones/rnabpflow/, with the infernal conda env on PATH for 'run'):
  python c1_rfam.py prep    --bundle-meta bundle/meta.json --c3 c3_facts.json \
                            --chains /data/.../rna3db_2024-04-26_chains.tsv --out rfam_scan
  python c1_rfam.py run     --out rfam_scan --cm Rfam.cm --procs 100
  python c1_rfam.py collect --out rfam_scan
"""
import argparse, csv, glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WINDOW, OVERLAP, BATCH_NT = 3000, 300, 9000

def windows(seq):
    if len(seq) <= WINDOW: return [(0, seq)]
    out, s = [], 0
    while True:
        out.append((s, seq[s:s + WINDOW]))
        if s + WINDOW >= len(seq): break
        s += WINDOW - OVERLAP
    return out

def prep(a):
    out = Path(a.out); (out / "q").mkdir(parents=True, exist_ok=True)
    seqs = {}
    meta = json.load(open(a.bundle_meta))
    for t, m in meta.items(): seqs[f"T:{t}"] = m["pred_seq"]
    if a.c3 and Path(a.c3).exists():
        for o in json.load(open(a.c3)):
            if o.get("seq"): seqs[f"X:{o['target']}"] = o["seq"]
    n_train = 0
    with open(a.chains, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["split"] == "train_set":
                seqs[f"R:{r['chain_id']}"] = r["sequence"]; n_train += 1
    items = []
    for k, s in seqs.items():
        for st, w in windows(s.replace("T", "U")): items.append((f"{k}|{st}", w))
    items.sort(key=lambda x: -len(x[1]))
    batches, cur, nt = [], [], 0
    for name, w in items:
        if cur and nt + len(w) > BATCH_NT: batches.append(cur); cur, nt = [], 0
        cur.append((name, w)); nt += len(w)
    if cur: batches.append(cur)
    for i, b in enumerate(batches):
        with open(out / "q" / f"batch_{i:04d}.fa", "w") as f:
            for name, w in b: f.write(f">{name}\n{w}\n")
    json.dump({k: len(s) for k, s in seqs.items()}, open(out / "lengths.json", "w"))
    print(f"targets {sum(k.startswith('T:') for k in seqs)}, extended {sum(k.startswith('X:') for k in seqs)}, "
          f"training chains {n_train}; windows {len(items)}; batches {len(batches)}; total nt {sum(len(w) for _, w in items)}")

def run_one(args):
    fa, cm, outdir = args
    tbl = outdir / "tbl" / (fa.stem + ".tbl")
    if tbl.exists(): return fa.name, "skip"
    tmp = tbl.with_suffix(".part")
    cmd = ["cmscan", "--cpu", "1", "--cut_ga", "--rfam", "--nohmmonly", "--noali", "-o", "/dev/null",
           "--tblout", str(tmp), cm, str(fa)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return fa.name, "FAIL " + r.stderr[-200:]
    tmp.rename(tbl); return fa.name, "ok"

def run(a):
    out = Path(a.out); (out / "tbl").mkdir(exist_ok=True)
    fas = sorted((out / "q").glob("batch_*.fa")); print(f"{len(fas)} batches, {a.procs} parallel", flush=True)
    done = 0
    with ThreadPoolExecutor(a.procs) as ex:
        for name, st in ex.map(run_one, [(f, a.cm, out) for f in fas]):
            done += 1
            if not st.startswith(("ok", "skip")) or done % 10 == 0: print(f"  {done}/{len(fas)} {name} {st}", flush=True)

def parse_tbl(path):
    rows = []
    for line in open(path):
        if line.startswith("#") or not line.strip(): continue
        p = line.rstrip("\n").split(None, 17)
        if len(p) < 17: continue
        rows.append(dict(model=p[0], acc=p[1], query=p[2], seq_from=int(p[7]), seq_to=int(p[8]), strand=p[9],
                         score=float(p[14]), evalue=float(p[15]), inc=p[16], desc=p[17] if len(p) > 17 else ""))
    return rows

def collect(a):
    out = Path(a.out); L = json.load(open(out / "lengths.json")); res = {}
    for t in sorted((out / "tbl").glob("*.tbl")):
        for h in parse_tbl(t):
            key, st = h["query"].rsplit("|", 1); st = int(st)
            lo, hi = sorted((h["seq_from"], h["seq_to"]))
            h["lo"], h["hi"] = lo + st, hi + st          # position in the full sequence (1-based)
            res.setdefault(key, []).append(h)
    fam = {}
    for key, n in L.items():
        hits = sorted(res.get(key, []), key=lambda h: -h["score"])
        top = hits[0] if hits else None
        fam[key] = dict(length=n, n_hits=len(hits),
                        top=None if top is None else dict(acc=top["acc"], model=top["model"], score=top["score"],
                                                          evalue=top["evalue"], coverage=round((top["hi"] - top["lo"] + 1) / n, 3)),
                        families=sorted({h["acc"] for h in hits}))
    json.dump(fam, open(out / "families.json", "w"))
    for pre, lab in (("T:", "targets"), ("X:", "extended cohort"), ("R:", "training chains")):
        ks = [k for k in fam if k.startswith(pre)]
        print(f"{lab}: {len(ks)} sequences, with >=1 Rfam hit: {sum(fam[k]['n_hits'] > 0 for k in ks)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["prep", "run", "collect"])
    ap.add_argument("--out", required=True); ap.add_argument("--bundle-meta"); ap.add_argument("--c3")
    ap.add_argument("--chains"); ap.add_argument("--cm", default="Rfam.cm"); ap.add_argument("--procs", type=int, default=64)
    a = ap.parse_args(); {"prep": prep, "run": run, "collect": collect}[a.phase](a)
