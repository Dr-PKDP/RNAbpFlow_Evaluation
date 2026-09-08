# RNA-UQ Data Pipeline

Data curation, featurization, and uncertainty-quantification evaluation
pipeline for RNA structure prediction, built on the RNAbpFlow backbone.

## Structure

```
manuscript/      Manuscript and Supplementary Material
data_curation/    Candidate filtering: freeze-date splitting, modified-
                  nucleotide exclusion, chain resolution, cohort
                  construction
featurization/    Converts PDB structures into RNAbpFlow's training
                  feature format
inference/        Inference input prep, MC-dropout sampling, ground-
                  truth retrieval
analysis/         Dispersion-error analysis, incremental-value testing,
                  conformal calibration, practical-value evaluation
patches/          Modifications to upstream RNAbpFlow training/
                  inference code
```

## Models and datasets

- RNAbpFlow (backbone structure-prediction model):
  https://github.com/Bhattacharya-Lab/RNAbpFlow
- RNA3DB training sequences (release `2024-04-26`, tag
  `incremental-update`): https://github.com/marcellszi/rna3db
- RCSB Protein Data Bank: https://www.rcsb.org
- MXfold2: https://github.com/mxfold/mxfold2
- SimFold (MultiRNAFold package): http://www.rnasoft.ca/download.html
- CONTRAfold: http://contra.stanford.edu/contrafold/
- FR3D-python: https://github.com/BGSU-RNA/fr3d-python

## Pipeline order

1. **data_curation/**
   `cutoff_feasibility.py` → `scan_modified_nucleotides.py` →
   `fetch_auth_chains.py` → `build_external_cohort.py`

2. **featurization/**
   `build_training_pkl.py`, `batch_build_training_set.py`

3. **patches/** — apply before inference; see `patches/README.md`

4. **inference/**
   `prep_external_inputs.py`, `run_mc_dropout_inference.py`,
   `download_ground_truth.py`

5. **analysis/**
   `analyze_dispersion_vs_error.py` → `dedupe_cohort.py` →
   `bootstrap_ci.py` → `nested_spline_model.py` +
   `paired_loocv_test.py` → `df_sensitivity.py`, `repeated_kfold.py`,
   `varying_coefficient_model.py` → `aurc_crossfitted.py` →
   `conformal_and_selective_risk.py`, `stratified_conformal.py`,
   `diagnose_conformal_width.py`

   `training_overlap_audit/`: `sequence_leakage_audit.py`,
   `cluster_and_grouped_cv.py`, `training_overlap_detailed.py`,
   `rna3db_train_sequences.json`

   `conformal_redesign/`: `three_way_split_conformal.py`,
   `bootstrap_width_diff.py`

   `additional_robustness/`: `repeated_cv_significance_test.py`
   (invalid — see docstring), `out_of_bootstrap_validation.py`,
   `generate_extra10_list.py` + `run_mc_dropout_inference_extra.py` +
   `convergence_analysis_pooled.py`, `aurc_robustness.py` +
   `aurc_bootstrap_ci.py`, `extract_gc_content.py`,
   `compare_excluded_included.py`

   Superseded, not used for reported conclusions:
   `check_length_confound.py`, `compare_selection_baseline.py`,
   `aurc_comparison.py`

## Requirements

See `requirements.txt`. Featurization requires the RNAbpFlow training
package (`nucleotide_constants`, `vocabulary`, dataset utilities)
importable from the working directory. Base-pair map generation
requires MXfold2, SimFold, and CONTRAfold installed separately (paths
configured in `inference/prep_external_inputs.py`). FR3D-python is
required for base-pair annotation during featurization.

## Outstanding

- Archived, versioned release with DOI. The repository itself is at
  https://github.com/Dr-PKDP/RNAbpFlow_Evaluation, now referenced in
  the manuscript's Data and Code Availability section; a Zenodo (or
  similar) archived release with a permanent DOI is still needed.
- `LICENSE` file
- RNAbpFlow checkpoint hash
- Consolidated target manifest (PDB ID, chain, release date, sequence,
  resolved-residue count, inclusion/exclusion reason, cluster ID,
  nearest training sequence, identity score)
