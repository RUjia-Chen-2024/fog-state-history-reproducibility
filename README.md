# FOG state-history reproducibility

Version **1.0.1** | [Versioned release](https://github.com/RUjia-Chen-2024/fog-state-history-reproducibility/releases/tag/v1.0.1)

Reproducibility materials for **State-history forecasting for pre-onset warning of gait disruptions from stepping force signals**.

This archive contains the frozen seed-2026 analysis of 21 stepping-force recordings from seven participants. Reference events are cycle-derived algorithmic gait disruptions, not independent clinical FOG annotations.

## What can be reproduced

- Retention-coordinate and saved-forecast checks: 21 recordings, 385,035 anchor-horizon pairs, and 88 test reference onsets.
- The primary inter-event evaluation and paired participant-cluster summaries for state-history, persistence, and classical comparators.
- The bounded 2T robustness check on the same issued activations.
- The warning-comparison, recorded-example, and simplified workflow figures.

The included forecasting implementation is available for inspection. Complete raw force exports and trained checkpoints are **not bundled**; the executable reproduction workflow begins with saved derived coordinates and original predictions. The full framework and forecast-MAE figures are retained outputs.

## Reproduce from saved outputs

From the repository root, using Python with the listed dependencies:

```bash
python -m pip install -r requirements.txt
python scripts/verify_derived.py
python scripts/rescore_inter_event.py
python scripts/recompute_strict_2T.py
python scripts/draw_inter_event_figures.py
```

No GPU or model retraining is required. Scripts regenerate tables and figures in their existing output directories. `MANIFEST_SHA256.json` describes the distributed files before any regeneration.

## Frozen evaluation

An activation is associated only with the next reference onset, and must fall strictly after the previous reference event ends and before that onset. No maximum lead is imposed, and the warning need not remain active until onset. The model's forecast horizon is approximately 1.5 s; association lead is a separate quantity.

| Predictor | Covered onsets | Associated activations / all activations | Median first-warning lead |
|---|---:|---:|---:|
| Persistence | 41/88 (46.6%) | 57/123 (46.3%) | 0.726 s |
| State-history | 71/88 (80.7%) | 117/194 (60.3%) | 0.936 s |

The secondary 2T check yields 37/88 (42.0%) and 67/88 (76.1%) coverage, respectively. The primary operator output includes 194 actual activations, 17.78 activations/min, and 42.7% active time. Operating points are not burden-matched. The endpoint and sensitivity were developed after earlier test inspection, not preregistered.

## Files

| Directory | Contents |
|---|---|
| `derived/coordinates/` | 21 full-length 100-Hz coordinate tables, compressed as CSV |
| `derived/predictions/` | 21 saved forecast tables, compressed as CSV |
| `derived/recordings.json` | Coded record IDs, split boundaries, thresholds and source hashes |
| `derived/reference_intervals.json` | Full-record algorithmic reference intervals |
| `analysis/inter_event_revision/` | Current primary results and original activation inputs |
| `analysis/strict_2T_sensitivity/` | Current bounded results |
| `analysis/classical_ews_revision/` | Historical comparator inputs; historical scores use earlier definitions |
| `analysis/classical_sensitivity/` | Comparator configuration inputs |
| `scripts/` | Portable verification, evaluation and plotting scripts |
| `implementation/` | Frozen model, coordinate and warning implementation |
| `provenance/` | Independent replay and derived-data verification records |
| `figures/` | Retained publication figures |

Use `analysis/inter_event_revision/` and `analysis/strict_2T_sensitivity/` for current outcomes. Earlier comparator files are retained as inputs and should not be substituted for the current evaluation.

## Data dictionary

Coordinate columns: `sample_index` is a zero-based original sample index; `time_sec` is recording time in seconds; `RA` and `RE` are dimensionless retention coordinates; `dlog_RA` and `dlog_RE` are backward log-slopes in s^-1.

Forecast columns: `anchor_index` and `target_index` use the same indexing; `anchor_time`, `target_time` and `horizon_sec` are seconds; `RA_pred` and `RE_pred` are raw-coordinate forecasts. Persistence equals the coordinates at the anchor. Observed future targets are coordinates at `target_index`.

Reference intervals are `[onset, end]` pairs in seconds. Test eligibility is defined by `test_start_time` in `recordings.json`.

## Data source and rights

The source data and processing scripts accompany Wang et al. (2023), *Time Series Analysis and Modeling of the Freezing of Gait Phenomenon*, DOI [10.1137/22M1484341](https://doi.org/10.1137/22M1484341). The authors' [Figshare resource](https://figshare.com/s/a14be7360925639736ba) states CC BY 4.0. The present archive provides derived coordinates, saved predictions and algorithmic labels; it is not the original full dataset. Cropped source force values used in the example figures are also included. See `THIRD_PARTY_NOTICES.txt`.

The archive creator is **Rujia Chen**. Cite the archive title, version 1.0.1, and the versioned release URL above. Version 1.0.1 adds creator and rights metadata for Zenodo archiving; all scientific data, predictions, evaluation scripts and results are unchanged from version 1.0.0.

No additional reuse license is granted for newly written study-specific code. Public access does not imply an open-source license. This does not change the original source-data CC BY 4.0 permissions. See `RIGHTS.txt` and `THIRD_PARTY_NOTICES.txt`.

The Zenodo DOI will be linked from the GitHub release after the archive service completes processing. `.zenodo.json` supplies the confirmed creator and rights information.
