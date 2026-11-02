"""
export_bundle.py -- one-pass exporter of the data used by the extended analyses.

Runs once, next to the predictions and ground-truth PDBs, and writes a small
bundle that the analysis scripts read. It does no inference and no statistics;
it only extracts what the extended analyses need and records how residues
were paired.

What goes into the bundle (per target, for the 82-target cohort):
  * C4' coordinates of every sample, for every run found (primary 20,
    dropout-off replicate 20, extra 10), over ALL predicted residues
  * C4' coordinates of the reference, the alignment index maps
    (pred_idx, gt_idx), author residue numbers of the paired residues
  * registration diagnostics: alignment mismatch rate, count of C4'-bearing
    HETATM residues, the C4' count the order-based v1 script would have
    seen, and whether v1 would have accepted the target
  * a re-computation of dispersion and error with the alignment-based logic, compared with
    the value already reported in the results JSON (reproduction check)
  * the raw RCSB metadata entry for each target (method, resolution, dates,
    entities) and an 82-sequence FASTA
  * optionally the three base-pair input maps (sparse) and an approximate
    reference base-pair list derived from the deposited atoms

Usage (from backbones/rnabpflow/ on the server):

  python export_bundle.py --check  <same arguments>      # counts only, no output
  python export_bundle.py \
      --results  dispersion_error_results_deduped.json \
      --cohort   external_cohort.json \
      --metadata rna_monomers_<DATE>.json \
      --gt-dir   ground_truth_pdbs \
      --run primary=Predictions \
      --run dropout_off=Predictions_dropout_off \
      --run extra10=Predictions_extra10 \
      --inputs-dir Inputs \
      --out bundle

--run may be repeated; a run whose directory is missing is reported and skipped.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from Bio import Align
from Bio.PDB import PDBParser, MMCIFParser

_NA = {"DA": "A", "DC": "C", "DG": "G", "DT": "U", "A": "A", "C": "C", "G": "G", "U": "U"}
MAX_MISMATCH_RATE = 0.05  # identical to analyze_dispersion_vs_error.py
BP_ATOMS = ("N1", "N3", "N2", "O2", "O4", "O6", "N4", "N6")


# ----------------------------------------------------------------- geometry
def kabsch_align(mobile, target):
    mc, tc = mobile - mobile.mean(0), target - target.mean(0)
    U, _, Vt = np.linalg.svd(mc.T @ tc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return (R @ mc.T).T + target.mean(0)


def rmsd(a, b):
    if a.shape != b.shape or a.shape[0] < 3:
        return None
    return float(np.sqrt(np.mean(np.sum((kabsch_align(a, b) - b) ** 2, 1))))


def disp_err(samples, gt):
    """Mean pairwise RMSD among samples and mean RMSD to gt (alignment-based definitions)."""
    pw = [rmsd(samples[i], samples[j]) for i in range(len(samples)) for j in range(i + 1, len(samples))]
    pw = [x for x in pw if x is not None]
    er = [rmsd(s, gt) for s in samples]
    er = [x for x in er if x is not None]
    return (float(np.mean(pw)) if pw else None, float(np.mean(er)) if er else None)


# --------------------------------------------------------------- structures
def parse_reference(pdb_path, chain_id):
    """Replicates get_gt_seq_and_coords and adds registration diagnostics."""
    parser = MMCIFParser(QUIET=True) if str(pdb_path).endswith(".cif") else PDBParser(QUIET=True)
    model = next(iter(parser.get_structure("x", str(pdb_path)).get_models()))
    chain = model[chain_id]
    seq, coords, resnums, atoms = [], [], [], []
    n_c4_all = n_het_c4 = n_atom_res = 0
    het_c4 = []
    nonstd_atom_res = []
    for res in chain:
        has_c4 = "C4'" in res
        if has_c4:
            n_c4_all += 1  # what the order-based v1 script counted
        if res.id[0] != " ":
            if has_c4:
                n_het_c4 += 1
                het_c4.append(f"{res.resname.strip()}{res.id[1]}")
            continue
        n_atom_res += 1
        name = res.resname.strip()
        if name not in _NA:
            if has_c4:
                nonstd_atom_res.append(f"{name}{res.id[1]}")
            continue
        if not has_c4:
            continue
        seq.append(_NA[name])
        coords.append(res["C4'"].coord)
        resnums.append(f"{res.id[1]}{res.id[2].strip()}")
        atoms.append({a: res[a].coord.astype(np.float32) for a in BP_ATOMS if a in res})
    diag = dict(n_c4_all_chain=n_c4_all, n_het_c4=n_het_c4, het_c4=het_c4,
                n_atom_residues=n_atom_res, nonstd_atom_residues_with_c4=nonstd_atom_res)
    return "".join(seq), np.array(coords, dtype=np.float32), resnums, atoms, diag


def read_pred(pdb_path):
    model = next(iter(PDBParser(QUIET=True).get_structure("x", str(pdb_path)).get_models()))
    return np.array([a.coord for a in model.get_atoms() if a.get_name() == "C4'"], dtype=np.float32)


def align_map(pred_seq, gt_seq):
    al = Align.PairwiseAligner()
    al.mode, al.match_score, al.mismatch_score = "global", 2, -1
    al.open_gap_score, al.extend_gap_score = -2, -0.5
    a = al.align(pred_seq, gt_seq)[0]
    ap, ag = a[0], a[1]
    pi, gi, n, mm, pp, gp = [], [], 0, 0, 0, 0
    for pc, gc in zip(ap, ag):
        if pc != "-" and gc != "-":
            n += 1
            mm += pc != gc
            pi.append(pp)
            gi.append(gp)
        pp += pc != "-"
        gp += gc != "-"
    return np.array(pi), np.array(gi), (mm / n if n else 1.0)


def approx_wc_pairs(atoms, seq, max_d=3.5, min_sep=4):
    """Approximate canonical/wobble pairs from deposited atoms (GC, AU, GU).
    A pair needs all its listed H-bond donor/acceptor distances <= max_d,
    except GC which needs >= 2 of 3. Index space = resolved-residue order.
    This is a screening annotation, not FR3D/DSSR; the plan labels it so."""
    def d(i, a, j, b):
        if a not in atoms[i] or b not in atoms[j]:
            return None
        return float(np.linalg.norm(atoms[i][a] - atoms[j][b]))

    rules = {("G", "C"): [("N1", "N3"), ("N2", "O2"), ("O6", "N4")],
             ("A", "U"): [("N6", "O4"), ("N1", "N3")],
             ("G", "U"): [("N1", "O2"), ("O6", "N3")]}
    out = []
    n = len(seq)
    for i in range(n):
        for j in range(i + min_sep, n):
            for (x, y), pairs in rules.items():
                for p, q in ((i, j), (j, i)):
                    if seq[p] == x and seq[q] == y:
                        ds = [d(p, a, q, b) for a, b in pairs]
                        ok = [v is not None and v <= max_d for v in ds]
                        need = 2 if (x, y) == ("G", "C") else len(pairs)
                        if sum(ok) >= need:
                            out.append((i, j))
    return np.array(sorted(set(out)), dtype=np.int32).reshape(-1, 2)


# ------------------------------------------------------------------ inputs
def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sequences_and_entries(metadata_path):
    data = load_json(metadata_path)
    seqs, entries = {}, {}
    for e in data["entries"]:
        rid = e.get("rcsb_id")
        entries[rid] = e
        for pe in e.get("polymer_entities", []) or []:
            ep = pe.get("entity_poly", {}) or {}
            if ep.get("rcsb_entity_polymer_type") == "RNA":
                s = ep.get("pdbx_seq_one_letter_code_can")
                if s:
                    seqs[rid] = s.replace("\n", "").upper()
    return seqs, entries


def target_list(results_path):
    data = load_json(results_path)
    if isinstance(data, dict):
        for k in ("results", "data", "targets"):
            if k in data:
                data = data[k]
                break
    ids = [d["target"] for d in data]
    reported = {d["target"]: d for d in data}
    return ids, reported


def find_gt(gt_dir, pdb_up):
    """PDB-format file if present, otherwise mmCIF (extended-cohort entries have no PDB file)."""
    for ext in (".pdb", ".cif"):
        p = Path(gt_dir) / f"{pdb_up}{ext}"
        if p.exists():
            return p
    return Path(gt_dir) / f"{pdb_up}.pdb"


def split_target(tid):
    pdb, chain = tid.split("_", 1)
    return pdb, chain


def sparse_map(path):
    m = np.load(path)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return None
    i, j = np.nonzero(m)
    return np.stack([i, j, m[i, j]], 1).astype(np.float32)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--run", action="append", default=[], help="name=dir (repeatable)")
    ap.add_argument("--inputs-dir", default=None)
    ap.add_argument("--out", default="bundle")
    ap.add_argument("--check", action="store_true", help="count files only; write nothing")
    ap.add_argument("--no-bp-ref", action="store_true")
    a = ap.parse_args()

    runs = {}
    for r in a.run:
        name, d = r.split("=", 1)
        runs[name] = Path(d)
    gt_dir = Path(a.gt_dir)
    targets, reported = target_list(a.results)
    seqs, entries = sequences_and_entries(a.metadata)
    cohort = {f"{c['pdb_id'].lower()}_{c['chain']}": c for c in load_json(a.cohort)}

    # ---- pre-flight (always printed; --check stops here)
    print(f"targets in results file: {len(targets)}")
    for name, d in runs.items():
        have = sum((d / t).is_dir() for t in targets)
        n_samp = [len(list((d / t).glob("Sample_*.pdb"))) for t in targets if (d / t).is_dir()]
        print(f"run {name:<12} dir={d}  target dirs found {have}/{len(targets)}  "
              f"samples/target min={min(n_samp) if n_samp else 0} max={max(n_samp) if n_samp else 0}")
    miss_gt = [t for t in targets if not find_gt(gt_dir, split_target(t)[0].upper()).exists()]
    miss_seq = [t for t in targets if split_target(t)[0].upper() not in seqs and split_target(t)[0] not in seqs]
    not_in_cohort = [t for t in targets if t not in cohort]
    print(f"missing ground-truth PDB: {len(miss_gt)} {miss_gt[:5]}")
    print(f"missing RNA sequence in metadata: {len(miss_seq)} {miss_seq[:5]}")
    print(f"targets absent from cohort json: {len(not_in_cohort)} {not_in_cohort[:5]}")
    if a.inputs_dir:
        have = sum(all((Path(a.inputs_dir) / t / f"map{k}.npy").exists() for k in (1, 2, 3)) for t in targets)
        print(f"base-pair maps present for {have}/{len(targets)} targets")
    if a.check:
        return

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arrays, meta, bp_arrays = {}, {}, {}
    t0 = time.time()
    for n, tid in enumerate(targets, 1):
        pdb, chain = split_target(tid)
        pdb_up = pdb.upper()
        pred_seq = seqs.get(pdb_up) or seqs.get(pdb)
        gt_path = find_gt(gt_dir, pdb_up)
        rec = dict(target=tid, pdb_id=pdb_up, chain=chain, status="ok")
        if pred_seq is None or not gt_path.exists():
            rec["status"] = "missing_input"
            meta[tid] = rec
            continue
        try:
            gt_seq, gt_c4, gt_resnums, atoms, diag = parse_reference(gt_path, chain)
            pi, gi, mm = align_map(pred_seq, gt_seq)
        except Exception as exc:  # recorded, never silent
            rec["status"] = f"reference_error: {exc}"
            meta[tid] = rec
            continue
        rec.update(diag)
        rec.update(n_entity=len(pred_seq), n_gt_resolved=len(gt_seq), n_paired=len(pi),
                   mismatch_rate=mm, alignment_ok=bool(mm <= MAX_MISMATCH_RATE and len(pi) >= 10),
                   v1_order_pairing_would_pass=bool(diag["n_c4_all_chain"] == len(pred_seq)),
                   gt_seq_equals_pred_seq=bool(gt_seq == pred_seq),
                   n_unpaired_gt=int(len(gt_seq) - len(pi)), pred_seq=pred_seq, gt_seq=gt_seq,
                   gt_resnums_paired=[gt_resnums[k] for k in gi],
                   reported=reported.get(tid, {}))
        arrays[f"{tid}__gt"] = gt_c4
        arrays[f"{tid}__pred_idx"] = pi.astype(np.int32)
        arrays[f"{tid}__gt_idx"] = gi.astype(np.int32)

        rec["runs"] = {}
        for name, d in runs.items():
            files = sorted((d / tid).glob("Sample_*.pdb")) if (d / tid).is_dir() else []
            if len(files) < 2:
                rec["runs"][name] = dict(n_samples=len(files))
                continue
            S = np.stack([read_pred(f) for f in files])  # (k, L_pred, 3)
            arrays[f"{tid}__{name}"] = S
            info = dict(n_samples=len(files), files=[f.name for f in files],
                        shape_ok=bool(S.shape[1] == len(pred_seq)))
            if info["shape_ok"] and rec["alignment_ok"]:
                dd, ee = disp_err(list(S[:, pi]), gt_c4[gi])
                info.update(D_recomputed=dd, E_recomputed=ee)
            rec["runs"][name] = info
        prim = rec["runs"].get("primary", {})
        if "D_recomputed" in prim and rec["reported"].get("dispersion") is not None:
            rec["reproduces_reported"] = bool(
                abs(prim["D_recomputed"] - rec["reported"].get("dispersion", np.nan)) < 1e-3 and
                abs(prim["E_recomputed"] - rec["reported"].get("error", np.nan)) < 1e-3)

        if not a.no_bp_ref and len(gi):
            ref_pairs = approx_wc_pairs([atoms[k] for k in range(len(atoms))], gt_seq)
            # remap resolved-order indices to predicted-residue indices
            g2p = {int(g): int(p) for g, p in zip(gi, pi)}
            keep = [(g2p[i], g2p[j]) for i, j in ref_pairs if i in g2p and j in g2p]
            bp_arrays[f"{tid}__ref_pairs"] = np.array(keep, dtype=np.int32).reshape(-1, 2)
        if a.inputs_dir:
            for k in (1, 2, 3):
                p = Path(a.inputs_dir) / tid / f"map{k}.npy"
                if p.exists():
                    sm = sparse_map(p)
                    if sm is not None:
                        bp_arrays[f"{tid}__map{k}"] = sm
        e = entries.get(pdb_up) or entries.get(pdb)
        if e is not None:
            rec["rcsb_entry"] = e
        meta[tid] = rec
        if n % 10 == 0:
            print(f"  {n}/{len(targets)} done ({time.time() - t0:.0f}s)", flush=True)

    np.savez_compressed(out / "coords.npz", **arrays)
    if bp_arrays:
        np.savez_compressed(out / "bp.npz", **bp_arrays)
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f)
    with open(out / "targets82.fasta", "w") as f:
        for t, r in meta.items():
            if "pred_seq" in r:
                f.write(f">{t}\n{r['pred_seq']}\n")

    ok = [r for r in meta.values() if r["status"] == "ok"]
    print("\n===== SUMMARY =====")
    print(f"exported {len(ok)}/{len(targets)} targets  ->  {out}")
    print(f"alignment_ok: {sum(r['alignment_ok'] for r in ok)}")
    print(f"v1 order-based pairing would have accepted: {sum(r['v1_order_pairing_would_pass'] for r in ok)}")
    print(f"targets with C4'-bearing HETATM residues: {sum(r['n_het_c4'] > 0 for r in ok)}")
    print(f"targets with unpaired reference residues: {sum(r['n_unpaired_gt'] > 0 for r in ok)}")
    rep = [r.get('reproduces_reported') for r in ok if 'reproduces_reported' in r]
    print(f"reproduces reported D and E: {sum(bool(x) for x in rep)}/{len(rep)}")
    bad = [r["target"] for r in ok if r.get("reproduces_reported") is False]
    if bad:
        print("  NOT reproduced:", bad)
    print("Next: tar czf bundle.tgz", out, "&& scp it down")


if __name__ == "__main__":
    sys.exit(main())
