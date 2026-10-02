# Concept-Based Explainability in Medical Image Classification

Code for the Master's thesis *Concept-Based Explainability in Medical Image Classification* (Cognitive Science, Universität Osnabrück) by Shruti Kakkar.

The thesis audits the validity of Visual-TCAV concept attributions for a ResNet50V2 classifier trained on ISIC 2019, with concepts derived from the derm7pt Seven-Point Checklist and two synthetic artifact concepts (`ruler_present`, `vignette_present`). It answers four research questions: layer validity (RQ1), internal consistency (RQ2), construct validity (RQ3) and reliability (RQ4).

The code builds on the original Visual-TCAV implementation by De Santis et al. (2024): https://github.com/DataSciencePolimi/Visual-TCAV

## Branches

- `main` holds the code as used for the submitted thesis. The tag `thesis_submission` marks that state.
- The remaining branches hold the development history of individual experiments.

## Repository layout

| Path | Contents |
|---|---|
| `src/VisualTCAV.py` | Adapted Visual-TCAV engine (padding-preserving resize, hard-negative CAVs, emblem check, seeded CAV runs) |
| `src/` | Data preparation and the baseline training script |
| `src2/` | Padding-preserving model, all thesis experiments and analysis scripts |
| `outputs2/` | Results written by the `src2/` scripts |
| `outputs/` | Results of the baseline training runs |
| `labels/` | Hand labels for the ruler-sorted image sets |
| `environment.yml` | Conda environment |

Datasets, trained models, concept image folders and caches are not tracked. The steps below recreate them.

## Setup

```bash
conda env create -f environment.yml
conda activate vtcav_derma
```

The environment uses Python 3.10 and TensorFlow 2.15. All experiments ran on a single NVIDIA GTX TITAN X.

## Data

Download the following and place them under `datasets/`:

- ISIC 2019 training images and ground truth: https://challenge.isic-archive.com/data/#2019
- ISIC 2019 test images and ground truth (same page)
- derm7pt `release_v0`: https://derm.cs.sfu.ca/

```
datasets/
├── ISIC_2019_Training_Input/ISIC_2019_Training_Input/*.jpg
├── ISIC_2019_Training_GroundTruth.csv
├── ISIC_2019_Test_Input/ISIC_2019_Test_Input/*.jpg
├── ISIC_2019_Test_GroundTruth.csv
└── release_v0/
    ├── images/
    └── meta/{meta.csv, train_indexes.csv, ...}
```

## Running the pipeline

Run every script from the repository root, for example `python src2/concept_set_sizes.py`. Later steps read the outputs and CAV caches of earlier ones.

### 1. Data preparation

```bash
python src/organise_test_images.py            # datasets/test_images_by_class/
python src/prepare_concepts.py                # concept_images/ (10 derm7pt concepts + 500 random images)
python src2/restore_ruler_sorted_folders.py   # datasets/ruler_sorted/ and datasets/ruler_sorted_test/ from labels/
```

### 2. Classifier

```bash
python src/train.py                           # direct-resize baseline; set RUN_TAG for each of the five runs
python src2/train_new_padOnly.py              # padding-preserving configuration, first run
python src2/train_new_padonly_multiseed.py --seed 2
python src2/train_new_padonly_multiseed.py --seed 3
```

Every Visual-TCAV analysis uses `models2/resnet50v2_isic2019_final_padonly_seed2.keras`. The trained model is attached to the [`thesis_submission` release](https://github.com/Shruti-Kakkar/Thesis_Experiment/releases/tag/thesis_submission). Place it at that path to skip training:

```bash
mkdir -p models2
wget -P models2 https://github.com/Shruti-Kakkar/Thesis_Experiment/releases/download/thesis_submission/resnet50v2_isic2019_final_padonly_seed2.keras
sha256sum models2/resnet50v2_isic2019_final_padonly_seed2.keras
# 010f490034df8c99e91e50bc1179c59a929f02c706f9742f87a2eabe6960cb6b
```

### 3. Clinical concepts (RQ1, RQ2, RQ4)

```bash
python src2/VisualTCAV_derma_global_padonly_seed2.py                   # hard-negative CAVs
python src2/VisualTCAV_derma_ttest_padonly_seed2_hardneg_randomsample.py
python src2/VisualTCAV_derma_ttest_padonly_seed2_hardneg.py
python src2/VisualTCAV_derma_local_padonly_seed2_hardneg.py
python src2/rq1_rq2_zero_attribution_tables.py
python src2/rq1_class_concentration_index.py
python src2/plot_rq1_layer_barchart.py
python src2/rq4_concept_size_correlation.py
python src2/concept_set_sizes.py
python src2/check_cav_negative_pool_sampling.py

# Calibration, once per concept
for c in streaks_regular pigment_network_typical streaks_irregular pigmentation \
         dots_and_globules_regular pigment_network_atypical regression_structures \
         dots_and_globules_irregular; do
    python src2/concept_calibration.py --concept $c
done
python src2/bwv_calibration.py
python src2/bwv_posthoc_addon.py
python src2/recompute_bwv_stats.py
python src2/rq4_bh_correction.py
```

### 4. Artifact concepts (RQ3, RQ4)

```bash
python src2/build_ruler_matched_concepts.py       # uses ruler_overlay.py
python src2/build_vignette_matched_concepts.py    # uses vignette_overlay.py
python src2/build_ruler_category_splits.py
python src2/VisualTCAV_derma_global_ruler_bias_matched.py
python src2/VisualTCAV_derma_global_vignette_matched.py
python src2/rq3_artifact_cav_cosine_similarity.py

for c in thick_edge lesion_side ticks_edge short_ruler; do
    python src2/VisualTCAV_derma_global_ruler_bias_by_category.py --category $c --cav-seed 42
done
python src2/VisualTCAV_derma_global_ruler_bias_by_category.py --category short_ruler --cav-seed 123
python src2/VisualTCAV_derma_global_rulerbias_vignette_randomsample.py

python src2/extract_global_vtcav_test_images.py   # lists the 200 MEL + 200 NV images labelled in labels/ruler_sorted_test.csv
python src2/ruler_bias_ground_truth_verification_matched.py
python src2/ruler_bias_ground_truth_finalize_matched.py
python src2/ruler_bias_top_attribution_heatmaps_matched.py

python src2/ruler_bias_counterfactual_test.py
python src2/ruler_bias_counterfactual_negcontrol_test.py
python src2/vignette_bias_counterfactual_test.py
python src2/vignette_bias_counterfactual_negcontrol_test.py
python src2/rq3_negcontrol_effect_sizes.py

python src2/ruler_bias_local_screening_matched.py
python src2/ruler_bias_finalize_matched.py
python src2/vignette_bias_local_screening_matched.py
python src2/vignette_bias_finalize_matched.py
python src2/vignette_padding_confound_test.py
python src2/plot_sampling_bug_comparison.py
python src2/rq4_square_image_rates.py
python src2/artifact_pool_coverage.py
```

### 5. Figures

```bash
python src2/concept_gallery_figure.py
python src2/regenerate_local_heatmap_figures_larger_fonts.py
python src2/plot_confusion_matrix.py
```

Figure scripts write to `outputs2/thesis_figures/` unless noted otherwise below.

## Thesis tables and figures

| Thesis | Script | Output |
|---|---|---|
| Section 4.2.2, padding ablation | `src/train.py`, `src2/train_new_padOnly.py`, `src2/train_new_padonly_multiseed.py` | `outputs/classification_report*.txt`, `outputs2/classification_report_PadOnly*.txt` |
| Figure 4.1, Figure 4.2, Figure 4.7, Figure 5.3 | `src2/concept_gallery_figure.py` | `outputs2/thesis_figures/` |
| Figure 4.4, Figure 4.5 | `src2/VisualTCAV_derma_local_padonly_seed2_hardneg.py`, `src2/regenerate_local_heatmap_figures_larger_fonts.py` | `outputs2/thesis_figures/local_heatmap_*.png` |
| Table 5.1 | `src2/concept_set_sizes.py` | `outputs2/concept_set_sizes.json` |
| Table 5.2 | `src2/train_new_padonly_multiseed.py --seed 2` | `outputs2/classification_report_PadOnly_seed2.txt` |
| Figure 5.1 | `src2/plot_confusion_matrix.py` | `outputs2/confusion_matrix_seed2.png` |
| Table 5.3, Table 5.4, Section 5.4 | `src2/rq1_rq2_zero_attribution_tables.py` | `outputs2/rq1_rq2_zero_attribution_tables.json` |
| Table 5.3 footnote | `src2/VisualTCAV_derma_ttest_padonly_seed2_hardneg.py` | `outputs2/vtcav_ttest_proper_ig_padonly_seed2_hardneg/ttest_results_proper_ig.json` |
| Figure 5.2 | `src2/plot_rq1_layer_barchart.py` | `outputs2/thesis_figures/rq1_layer_barchart_MEL.png` |
| Section 5.3, class-concentration index | `src2/rq1_class_concentration_index.py` | printed |
| Table 5.5, validation accuracy | `src2/VisualTCAV_derma_global_ruler_bias_matched.py`, `src2/VisualTCAV_derma_global_vignette_matched.py` | printed |
| Table 5.5, cosine similarity | `src2/rq3_artifact_cav_cosine_similarity.py` | `outputs2/rq3_artifact_cav_cosine_similarity.json` |
| Section 5.5.1, per-style validation accuracy and seed check | `src2/VisualTCAV_derma_global_ruler_bias_by_category.py` | `outputs2/vtcav_results_padonly_seed2_rulerbias_matched_<style>_cavseed<N>/results.json` |
| Table 5.6, Table 5.7 | `src2/VisualTCAV_derma_global_rulerbias_vignette_randomsample.py` | `outputs2/global_attribution_randomsample_check/results.json` |
| Table 5.8 | four `*_counterfactual_*test.py` scripts, `src2/rq3_negcontrol_effect_sizes.py` | `outputs2/rq3_negcontrol_effect_sizes.json` |
| Figure 5.4, Figure 5.5 | `src2/ruler_bias_counterfactual_test.py` | `outputs2/ruler_bias_counterfactual_test_set/` (`example_images/pair_ISIC_0034324.png`, `counterfactual_shift_plot.png`) |
| Figure 5.6, Figure 5.7 | `src2/vignette_bias_counterfactual_test.py` | `outputs2/vignette_bias_counterfactual_test_set/` (`example_images/pair_ISIC_0034330.png`, `counterfactual_shift_plot.png`) |
| Table 5.9 | `src2/ruler_bias_ground_truth_verification_matched.py`, `src2/ruler_bias_ground_truth_finalize_matched.py` | `outputs2/ruler_bias_ground_truth_verification_matched/` |
| Figure 5.8 | `src2/ruler_bias_top_attribution_heatmaps_matched.py`, `src2/regenerate_local_heatmap_figures_larger_fonts.py` | `outputs2/thesis_figures/ruler_ground_truth_heatmap.png` |
| Figure 5.9 | screening scripts, `src2/plot_sampling_bug_comparison.py` | `outputs2/sampling_bug_comparison.png` |
| Section 5.6.1, square-image rates | `src2/rq4_square_image_rates.py` | `outputs2/rq4_square_image_rates.json` |
| Table 5.10 | `src2/vignette_padding_confound_test.py` | `outputs2/vignette_bias_local_screening_matched/padding_confound_square_attributions.csv` |
| Section 5.6.2 | `src2/rq4_concept_size_correlation.py` | printed |
| Table 5.11 | `src2/concept_calibration.py`, `src2/bwv_calibration.py`, `src2/bwv_posthoc_addon.py`, `src2/recompute_bwv_stats.py`, `src2/rq4_bh_correction.py` | `outputs2/<concept>_calibration/`, `outputs2/bwv_calibration/`, `outputs2/concept_calibration_bh_correction.csv` |
| Figure 5.10 | `src2/bwv_calibration.py` | `outputs2/bwv_calibration/bwv_calibration_plot.png` |
| Section 5.6.4, Fisher's exact test | `src2/concept_calibration.py`, `src2/bwv_posthoc_addon.py` | calibration summaries |
| Limitations, negative-pool sampling | `src2/check_cav_negative_pool_sampling.py` | `outputs2/cav_negative_pool_sampling_check/results.json` |
| Limitations, artifact image pool | `src2/artifact_pool_coverage.py` | `outputs2/artifact_pool_coverage.json` |

The `short_ruler` row of Table 5.6 uses the CAV trained with `--cav-seed 123`. The other three styles use `--cav-seed 42`.

The biased-sample results in Figure 5.9 (`outputs2/*_local_screening_matched_biased_sample_archive/`) come from the earlier, `files[:n]` version of the screening scripts in the git history.

## Superseded scripts

These scripts are not used for any result in the thesis. Each file starts with a `SUPERSEDED` header.

| Script | Replaced by |
|---|---|
| `src/VisualTCAV_derma_global.py` | `src2/VisualTCAV_derma_global_padonly_seed2.py` |
| `src/VisualTCAV_derma_ttest.py` | `src2/VisualTCAV_derma_ttest_padonly_seed2_hardneg_randomsample.py` |
| `src2/train_new.py` | `src2/train_new_padOnly.py`, `src2/train_new_padonly_multiseed.py` |
| `src2/VisualTCAV_derma_ttest_padonly_seed2.py` | `src2/VisualTCAV_derma_ttest_padonly_seed2_hardneg_randomsample.py` |
| `src2/VisualTCAV_derma_ttest_padonly_seed2_hardneg_randomsample_5class_post_relu_check.py` | `src2/VisualTCAV_derma_ttest_padonly_seed2_hardneg_randomsample.py` |
| `src2/VisualTCAV_derma_global_ruler_bias.py` | `src2/VisualTCAV_derma_global_ruler_bias_matched.py` |
| `src2/VisualTCAV_derma_local_ruler_bias.py` | `src2/ruler_bias_ground_truth_verification_matched.py`, `src2/ruler_bias_top_attribution_heatmaps_matched.py` |
| `src2/VisualTCAV_derma_local_ruler_bias_matched.py` | `src2/ruler_bias_ground_truth_verification_matched.py`, `src2/ruler_bias_top_attribution_heatmaps_matched.py` |
| `src2/ruler_bias_common.py` | `src2/ruler_bias_common_matched.py` |
| `src2/ruler_bias_finalize.py` | `src2/ruler_bias_finalize_matched.py` |
| `src2/ruler_bias_local_screening.py` | `src2/ruler_bias_local_screening_matched.py` |
| `src2/regenerate_local_heatmap_figures.py` | `src2/regenerate_local_heatmap_figures_larger_fonts.py` |
| `src2/ruler_prevalence_by_class.py` | exploratory data check |
| `src2/sanity_check_resize.py` | one-off visual check |
| `src2/ruler_short_ruler_local_check.py` | diagnostic heatmap |
