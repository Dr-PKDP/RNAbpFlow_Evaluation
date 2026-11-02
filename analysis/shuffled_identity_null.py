"""
shuffled_identity_null.py -- calibrates the training-overlap identity thresholds
(80% global; 93%/50% local fragment) against sequences known to be unrelated.

For each target's resolved sequence, K shuffled copies are generated (dinucleotide-
preserving, Altschul-Erickson/Jiang et al. 2008, and mononucleotide), and each copy is
passed through exactly the procedure of training_overlap_detailed.py: global
alignment against every RNA3DB training sequence (match 1, mismatch -1, gap open -2,
extend -0.5), identity = matches / max(len(query), len(train)), nearest chain by global
identity, then local alignment to that chain (identity = matches / aligned positions,
coverage of the query).  The real sequences are processed by the same code as a
reproduction check.

Usage:
  python shuffled_identity_null.py --targets targets.json --train rna3db_train_sequences.json \
         --out null.json --k-di 10 --k-mono 10 --procs 8 [--limit N] [--seed 1]
targets.json: {"target_id": "RESOLVED_SEQUENCE", ...}
"""
import argparse, json, random, time
from multiprocessing import Pool
from Bio.Align import PairwiseAligner

def mk(mode):
    a = PairwiseAligner(); a.mode = mode
    a.match_score = 1; a.mismatch_score = -1; a.open_gap_score = -2; a.extend_gap_score = -0.5
    return a

def detail(q, s, al):
    aln = al.align(q, s)[0]; a1, a2 = str(aln[0]), str(aln[1])
    m = sum(1 for x, y in zip(a1, a2) if x == y and x != '-')
    nb = sum(1 for x, y in zip(a1, a2) if x != '-' and y != '-')
    return m, nb

def dinuc_shuffle(seq, rng):
    """Dinucleotide-preserving shuffle (random Eulerian path; Altschul & Erickson 1985)."""
    if len(seq) < 3: return seq
    last = seq[-1]; edges = {}
    for a, b in zip(seq[:-1], seq[1:]): edges.setdefault(a, []).append(b)
    # choose a random last-edge for every vertex except the final one, forming an arborescence into `last`
    while True:
        lastedge = {}
        for v in edges:
            if v == last: continue
            lastedge[v] = rng.choice(edges[v])
        # check that following last edges from every vertex reaches `last`
        ok = True
        for v in lastedge:
            seen = set(); u = v
            while u != last:
                if u in seen or u not in lastedge: ok = False; break
                seen.add(u); u = lastedge[u]
            if not ok: break
        if ok: break
    pools = {}
    for v, lst in edges.items():
        l = list(lst)
        if v != last:
            l.remove(lastedge[v])
            rng.shuffle(l); l.append(lastedge[v])
        else:
            rng.shuffle(l)
        pools[v] = l
    out = [seq[0]]; v = seq[0]
    for _ in range(len(seq) - 1):
        v2 = pools[v].pop(0); out.append(v2); v = v2
    return "".join(out)

def mono_shuffle(seq, rng):
    l = list(seq); rng.shuffle(l); return "".join(l)

TRAIN = None
def init(train):
    global TRAIN; TRAIN = train

def work(job):
    tid, kind, k, q = job
    g, l = mk("global"), mk("local")
    best, bid = -1.0, None
    for cid, s in TRAIN.items():
        m, nb = detail(q, s, g); pid = m / max(len(q), len(s))
        if pid > best: best, bid = pid, cid
    s = TRAIN[bid]
    mg, nbg = detail(q, s, g); ml, nbl = detail(q, s, l)
    return dict(target=tid, kind=kind, rep=k, qlen=len(q), nearest=bid, global_id=round(best, 4),
                local_id=round(ml / nbl, 4) if nbl else 0.0, local_aln_len=nbl, local_cov_target=round(nbl / len(q), 4))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True); ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--k-di", type=int, default=10)
    ap.add_argument("--k-mono", type=int, default=10); ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    T = json.load(open(a.targets)); train = json.load(open(a.train))
    ids = sorted(T)[: a.limit] if a.limit else sorted(T)
    rng = random.Random(a.seed); jobs = []
    for t in ids:
        jobs.append((t, "real", 0, T[t]))
        for k in range(a.k_di): jobs.append((t, "dinuc", k, dinuc_shuffle(T[t], rng)))
        for k in range(a.k_mono): jobs.append((t, "mono", k, mono_shuffle(T[t], rng)))
    jobs.sort(key=lambda j: -len(j[3]))      # longest first for load balance
    print(f"{len(ids)} targets, {len(jobs)} queries, {a.procs} processes", flush=True)
    part = a.out + ".jsonl"; done = set(); res = []
    try:
        for line in open(part):
            r = json.loads(line); res.append(r); done.add((r["target"], r["kind"], r["rep"]))
    except FileNotFoundError:
        pass
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print(f"resuming: {len(done)} done, {len(jobs)} to go", flush=True)
    t0 = time.time()
    with Pool(a.procs, initializer=init, initargs=(train,)) as p, open(part, "a") as fp:
        for i, r in enumerate(p.imap_unordered(work, jobs, chunksize=1), 1):
            res.append(r); fp.write(json.dumps(r) + "\n"); fp.flush()
            if i % 25 == 0: print(f"  {i}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(a.out, "w")); print(f"done in {time.time()-t0:.0f}s -> {a.out}")
