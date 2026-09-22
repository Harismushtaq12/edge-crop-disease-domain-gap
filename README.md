# Lab-Accurate, Field-Blind

Code, saved run artefacts and figures for *Lab-Accurate, Field-Blind:
Measuring the Domain Gap in Edge Crop-Disease Models* (submitted to SN
Computer Science).

A MobileNetV2 that scores **78.7%** on PlantVillage laboratory tomato images
scores **12.1%** on PlantDoc field photographs of the same eight classes.
That is below the 20.3% a constant predictor achieves on the same set. This
repository contains everything behind that number and behind the six remedies
we tested against it.

## Start here

**[`notebooks/01_reproduce_paper_results.ipynb`](notebooks/01_reproduce_paper_results.ipynb)**
is already executed and renders on GitHub. It rebuilds every table and every
derived number in the paper from the saved result files, re-derives the
bootstrap intervals and the chance floors, hand-checks the memory arithmetic,
and displays all 17 figures. Reading it requires nothing; re-running it takes
a few seconds and needs only `numpy` and `pandas`.

It does **not** retrain. That is the point of the split: checking the paper's
arithmetic should not cost a day of CPU.

## What is in here

| Path | Contents |
|---|---|
| `notebooks/` | The executed results notebook, plus the script that builds it |
| `scripts/` | The full pipeline, numbered in run order (see below) |
| `results/` | Every per-run JSON and training log the paper cites |
| `figures/` | All 17 paper figures at submission resolution |
| `paper/` | The compiled manuscript |

## Reproducing from scratch

The numbered scripts run in order. Stages 01–06 are the main experiment;
everything above 07 is a remedy, an ablation or an analysis, and each reads
the artefacts of the stages before it.

```
01_build_manifest.py          index both datasets into manifest.csv
02_extract_features.py        (optional) feature cache
03_train.py                   from-scratch baseline, both scales
03b_train_arch.py             MobileNetV3-Small and EfficientNetB0
04_quantize_and_benchmark.py  INT8 export, accuracy and latency
05_per_class_analysis.py      per-class accuracy and confusion matrices
15_bootstrap_ci.py            bootstrap intervals on the gap

07_finetune_mitigation.py     few-shot: full fine-tune
07b_finetune_head_only.py     few-shot: head only
07c_finetune_l2sp.py          few-shot: L2-SP
25_cutmix_training.py         CutMix augmentation
28_convert_pretrained_weights.py  port the 96px ImageNet checkpoint
29_pretrained_finetune.py     ImageNet-pretrained arm
32_stacked_pretrained_finetune.py  pretrained + few-shot combination
33_resolution_ablation_224.py 224px resolution ablation

16_gradcam.py                 qualitative Grad-CAM
18_gradcam_bbox_energy.py     leaf-box attention metric, baseline
26/30/40/45_gradcam_*.py      the same metric for each remedy
54_gradcam_on_fttest.py       restrict every arm to the held-out split
20_layer_sensitivity_blocks.py  per-layer quantization sweep
47_batch_memory_probe.py      measured training memory vs batch size
52_peak_sram_tflite.py        activation arena on the converted INT8 graph
37_unified_scoreboard.py      re-score every arm on one basis
```

Figures are produced by `35`, `36`, `39`, `41`–`46` and `48`–`51`, all of
which import `figstyle.py`. The two schematics in the paper are TikZ and live
with the manuscript source, not here.

You will need to fetch PlantVillage and PlantDoc yourself; `01_build_manifest.py`
expects them on disk and writes the index the rest of the pipeline reads.

## Notes that will save you time

**Three scripts supersede earlier ones**, and the superseded versions are kept
only for provenance:

- `52_peak_sram_tflite.py` replaces `38_peak_sram_analysis.py`. The old script
  walked the Keras layer list, which keeps padding and activation layers that
  the TFLite converter fuses away, so it scored operators that never execute
  and put every backbone's peak on a padding layer. Reading the converted
  graph instead changed the numbers substantially (MobileNetV2 441 → 270 KB,
  MobileNetV3-Small 84 → 72 KB, EfficientNetB0 441 → 648 KB) and made each one
  checkable by hand from tensor shapes.
- `54_gradcam_on_fttest.py` restricts every attention number to `D_ft-test`.
  The adapted arms trained on 160 of the 744 field images, so scoring them
  over all 623 box-annotated images scored them partly on their own training
  data.
- `49`/`50`/`51_*_v3.py` replace `24`/`13`/`16` for the dataset-sample,
  prediction-grid and Grad-CAM figures.

**The batch-memory probe must instantiate the real input pipeline.** Fed
synthetic tensors it reports batch 64 surviving at 5.94 GB, where the real run
is killed at 5.97 GB. The missing 1.2 GB is the shuffle buffer.

**BatchNorm momentum.** Keras's default of 0.99 assumes many thousands of
steps. At the step counts here it leaves the running statistics near their
initialisation, and the model's evaluation-mode predictions collapse to one
constant class while training accuracy rises normally. `m=0.90` fixes it. The
symptom looks exactly like a data bug.

## Requirements

```
pip install -r requirements.txt
```

TensorFlow 2.21 on CPU is what everything was run on. The results notebook
alone needs only `numpy`, `pandas` and `jupyter`.

## Data

PlantVillage and PlantDoc are third-party datasets under their own licences,
and are not redistributed here. `results/manifest.csv` records which images
were used and how they were split, but contains no image data.

## Citation

The manuscript is under review. Citation details will be added on acceptance.
