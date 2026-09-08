# Featurization Notes

## Base-pair feature (`ss`) approximation

The training feature format includes a 3-channel base-pair matrix
(`ss`, duplicated in `custom_ss_mask`). Channel 0 corresponds exactly to
canonical Watson-Crick pairing (FR3D's `cWW` classification); this
reproduces the original training pipeline's output exactly.

Channels 1 and 2 in the original pipeline distinguish additional pairing
categories, but the exact definitions could not be recovered from the
released training artifacts. This pipeline substitutes FR3D's combined
canonical + non-canonical pairing classification (`basepair_detail`,
unfiltered) for both channels.

Empirical characteristics of this substitution, measured against
reference feature files where available:

- 100% precision: every base pair reported is a real, geometrically
  valid pairing interaction. No false pairs are introduced.
- Partial recall: approximately 79-95% of true pairs in the reference
  channels are recovered, depending on the target. The gap corresponds
  to pairing relationships not captured by any tested FR3D interaction
  category (canonical, non-canonical, near-threshold, or several other
  geometric relation types). Stacking annotations show no measurable
  overlap with the reference channels.

## Backbone-atom indexing

The compact 23-atom-per-residue representation used by this format
places different atoms at different array positions depending on
residue type in general, but the training pipeline's rigid-frame and
centering computations reference the atom at a *fixed* array index
(3) regardless of residue type. This index corresponds to a different
atom by name (`C2'`) than the one the pipeline's own full-length
27-atom table associates with that index (`C4'`). This featurizer
matches the fixed-index behavior exactly, since that is what downstream
consumers (rigid-frame extraction, coordinate centering) actually use.

## Modified nucleotides

Not supported. Structures containing them should be excluded from the
candidate pool prior to featurization (see
`data_curation/scan_modified_nucleotides.py`). Inosine is not detected
by that sequence-based filter and will instead cause an explicit,
named error during featurization.
