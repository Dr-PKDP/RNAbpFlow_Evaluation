# RNAbpFlow evaluation: ensemble dispersion and structural error

Code and cohort manifests for the evaluation of RNAbpFlow sample ensembles on a post-training-cutoff set of RNA structures. The pipeline builds the cohort, generates 20 samples per target, computes the ensemble dispersion D (mean pairwise C4′ RMSD) and the structural error E (mean C4′ RMSD to the deposited reference), and tests whether D improves the prediction of E beyond sequence length.

## Layout

```
data_curation/        Candidate filtering: release-date window, modified-
                      nucleotide exclusion, chain resolution, cohort
featurization/        RNAbpFlow training-feature construction
inference/            Base-pair map preparation, sampling, reference retrieval
patches/              Required changes to upstream RNAbpFlow code
stochasticity_control/
                      Checkpoint audit and the dropout-off / dropout-only controls
analysis/             Statistical analyses
analysis/extended/    Coupling, independent-ensemble, ensemble-size, registration,
                      alternative-metric, nonlinear and varying-coefficient analyses
manifest/             Cohort manifests and the script that builds them
```

## External resources

- RNAbpFlow: https://github.com/Bhattacharya-Lab/RNAbpFlow
- RNA3DB (release `2024-04-26`, tag `incremental-update`): https://github.com/marcellszi/rna3db
- RCSB Protein Data Bank: https://www.rcsb.org
- MXfold2: https://github.com/mxfold/mxfold2
- SimFold (MultiRNAFold package): http://www.rnasoft.ca/download.html
- CONTRAfold: http://contra.stanford.edu/contrafold/
- FR3D-python: https://github.com/BGSU-RNA/fr3d-python
- Infernal 1.1.4 and Rfam 14.10, for the family assignments

Python dependencies are in `requirements.txt`. Featurization needs the RNAbpFlow training package (`nucleotide_constants`, `vocabulary`, dataset utilities) on the import path, and FR3D-python for base-pair annotation. The scripts in `analysis/extended/`, `analysis/aurc_*.py` and `manifest/` read their inputs from a data directory given by the environment variable `RNA_DATA` (default `data`), which holds `bundle/` (written by `analysis/export_bundle.py`), `small/` (the per-target result files written by the analysis scripts), `c2/` and `rf/` (shuffled-identity and Rfam outputs).

## Pipeline

1. **data_curation/**: `cutoff_feasibility.py`, `scan_modified_nucleotides.py`, `fetch_auth_chains.py`, `build_external_cohort.py`.
2. **featurization/**: `build_training_pkl.py`, `batch_build_training_set.py`.
3. **patches/**: apply `apply_datamodule_seed_patch.py` and `apply_dataset_seed_patch.py` before training or inference (see below).
4. **inference/**: `prep_external_inputs.py` (writes `map1.npy` to `map3.npy` per target with the base-pair map generator, which runs MXfold2, SimFold and CONTRAfold; its path is set at the top of the script), `run_mc_dropout_inference.py` (20 samples per target), `download_ground_truth.py`.
5. **analysis/**:
   - Dispersion and error: `analyze_dispersion_vs_error.py` (residue pairing by sequence alignment), then `dedupe_cohort.py` (exact-sequence deduplication; also writes `duplicate_of.json`).
   - Primary comparison: `bootstrap_ci.py`, `nested_spline_model.py`, `paired_loocv_test.py`, `verify_loocv_knot_leakage.py`, `extract_paired_diffs.py`, `df_sensitivity.py`, `repeated_kfold.py`.
   - Varying-coefficient model: `varying_coefficient_model.py` and the fold-safe version `varying_coefficient_model_foldsafe.py`.
   - Selective risk and conformal prediction: `aurc_comparison.py`, `aurc_crossfitted.py`, `aurc_table5.py`, `aurc_final.py`, `conformal_and_selective_risk.py`, `stratified_conformal.py`, `residual_conformal.py`, `diagnose_conformal_width.py`; `conformal_redesign/` holds `three_way_split_conformal.py` and `bootstrap_width_diff.py` (paired width bootstrap: both models are fitted and evaluated on the same partitions within each resample).
   - Training overlap and families: `training_overlap_audit/` (`sequence_leakage_audit.py`, `training_overlap_detailed.py`, `cluster_and_grouped_cv.py`, `rna3db_train_sequences.json`), `shuffled_identity_null.py`, `c1_rfam.py`.
   - Ensemble size and independence: `convergence_analysis.py`, `compare_dropout_variance.py`, and in `additional_robustness/` `generate_extra10_list.py`, `run_mc_dropout_inference_extra.py`, `convergence_analysis_pooled.py`.
   - Further robustness: `additional_robustness/` also has `out_of_bootstrap_validation.py` (B = 500, 1,000, 2,000 and 4,000; reported at 4,000), `aurc_robustness.py`, `aurc_bootstrap_ci.py`, `extract_gc_content.py`, `compare_excluded_included.py`; `c3_get_cif.py` retrieves mmCIF files for entries without PDB-format coordinates; `export_bundle.py` writes the data bundle read by `analysis/extended/`.
   - `analysis/extended/`: `a1_a5.py`, `a2_a5.py`, `a6_a7.py`, `a8_a9.py`, `a10.py`, `a11.py`, `a12.py` and `a13_a14.py` (coupling, independent-ensemble, ensemble-size, registration, alternative-metric, resolved-coverage, extended-cohort and input-map analyses), `c1_analysis.py` and `c2_analysis.py` (Rfam families, shuffled-identity null), `n82.py` (GC content and identity on all 82 targets), `vc_df.py` and `nonlinD.py` (varying-coefficient model and nonlinear log D at 3 to 7 degrees of freedom), `a14_tm_dispersion.py` (TM-score outcome with dispersion in the same metric) and `a15_method_resolution.py` (structure-determination method and resolution), `pairwise_matrices.py` (pooled 50-sample RMSD matrices read by `a8_a9.py` and `a10.py`), `bund.py`, `lib.py` and `metrics.py` (shared loaders; C4′ TM-score, lDDT and contact F1).
6. **manifest/**: `build_manifest.py` writes the two files below.

## Cohort manifests

`manifest/manifest_82_targets.csv` has one row per evaluation target: release date, method, resolved residues and entity length, sequence cluster, D, E, leave-one-out predictions of Models 0 and 1, global identity and audit flags.

`manifest/candidates_196_entries.csv` gives the status of all 196 candidates (82 retained, 69 excluded as exact-sequence duplicates, 45 excluded because only mmCIF coordinates exist). For each duplicate, `duplicate_of` names the retained target with the same RNA sequence; the 69 duplicates map to 23 retained targets. The retained target of a sequence group is its alphabetically first member.

## Terminology and legacy names

The ensemble spread comes from the sampler: every dropout module in the released checkpoint has p = 0. Several scripts and output folders keep the earlier "dropout" naming without any change in behaviour.

| Term | Meaning | Legacy names |
|---|---|---|
| Sampling stochasticity | Centred Gaussian translation and uniform SO(3) rotation per sample, no fixed seed | `mc_dropout`, `Predictions/`, `Sample_*.pdb` |
| Dropout | The network's dropout layers (all p = 0) | `enable_mc_dropout()`, `dropout_on`, `dropout_off` |
| Ensemble dispersion (D) | Mean pairwise C4′ RMSD over the 20 samples | `dispersion` |
| Structural error (E) | Mean C4′ RMSD to the reference over resolved residues | `error`, `mean_rmsd_to_gt` |
| Primary run | The 20 samples behind every headline result | `run_mc_dropout_inference.py` |
| Replicate run | A second 20-sample run, dropout in its default (off) state | `stochasticity_control/run_generative_only_inference.py`, `Predictions_dropout_off/` |
| Extra samples | Ten further samples per target | `run_mc_dropout_inference_extra.py`, `Predictions_extra10/` |

`run_mc_dropout_inference.py` switches the dropout layers to train mode, which has no effect at p = 0. `stochasticity_control/` holds the checkpoint-metadata inspection, the per-module audit and reproducibility test (`freeze_dropout_evidence.py`), the dropout-off control (Condition A, `run_generative_only_inference.py`) and the dropout-only control (Condition C, initial frame fixed, `run_dropout_only_inference.py` with `patch_interpolant_scoped_seed.py`).

Seeds: the target bootstrap uses seed 0 (2,000 resamples for the AURC), the cluster bootstrap seed 1 (4,000 resamples), and the primary LOOCV bootstrap 10,000 target resamples. Random AURC reference values average 20,000 random orderings.

## Patches to upstream code

1. **Training data loader** (`src/data/pdb_na_datamodule_base.py`, `train_dataloader`): the length batcher is built without a seed and falls back to its class default, so every training run sees the same data order. The patch passes `getattr(self.data_cfg, "seed", 123)` as `seed`; set a distinct seed per run with the Hydra override `+data_cfg.seed=<value>`.
2. **Inference dataset** (`src/data/dataset.py`, `RNADataset.__init__`): each sample is seeded as `start_seed * (sample_id + 1) * 10`, which makes runs byte-identical for the same sample index. The patch sets `seedval = -1`, which the sampling code treats as "do not seed". This removes exact reproducibility of the shipped example outputs; revert it to reproduce those.

## Featurization notes

- **Base-pair feature (`ss`)**: channel 0 is canonical Watson-Crick pairing (FR3D `cWW`) and reproduces the original pipeline exactly. The definitions of channels 1 and 2 could not be recovered from the released training artifacts, so both use FR3D's combined canonical and non-canonical classification (`basepair_detail`, unfiltered). Measured against reference feature files, precision is 100% and recall is about 79 to 95% depending on the target.
- **Backbone-atom indexing**: the 23-atom residue format places atoms at residue-dependent positions, but the training pipeline's rigid-frame and centring computations read a fixed index (3), which is `C2'` by name where the pipeline's 27-atom table gives `C4'`. The featurizer reproduces the fixed-index behaviour, since downstream code uses it.
- **Modified nucleotides** are not supported and must be removed beforehand (`data_curation/scan_modified_nucleotides.py`). Inosine is not caught by that sequence-based filter and raises an explicit error during featurization.
