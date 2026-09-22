"""
Two-panel figure for the CutMix training-time-augmentation experiment
(25_cutmix_training.py + 26_cutmix_gradcam_bbox.py), added in response to
an external review's "mitigations tested are a strawman" critique: unlike
the three post-hoc field fine-tuning variants (Section IV-F/V-E), which
each use a small amount of REAL FIELD-LABELED data, CutMix uses none at
all -- it only changes how the lab-only training images are augmented.
Its field-accuracy comparison is therefore against the same 744-image
field set used in Section V-A (not the 584-image D_ft-test subset), since
no field images are held out or used at all for this variant.

Panel (a): lab vs. field accuracy, baseline (no CutMix) vs. CutMix.
Panel (b): quantitative Grad-CAM energy-in-leaf-box vs. chance, baseline
vs. CutMix -- the direct mechanistic test of whether forcing the model to
see occluded/patched leaves during training pulls its attention back
toward the leaf and away from background, the failure mode Section V-C
documents in the baseline model.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

baseline_acc = json.load(open("results_tomato_fp32.json"))
cutmix_acc = json.load(open("results_tomato_cutmix_fp32.json"))
baseline_gcam = json.load(open("gradcam_bbox_energy_results.json"))
cutmix_gcam = json.load(open("gradcam_bbox_energy_cutmix_results.json"))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

# --- panel (a): lab vs field accuracy ---
ax = axes[0]
labels = ["Lab\n(PlantVillage)", "Field\n(PlantDoc, n=744)"]
base_vals = [baseline_acc["pv_test"]["accuracy"] * 100, baseline_acc["pd_field"]["accuracy"] * 100]
cutmix_vals = [cutmix_acc["pv_test"]["accuracy"] * 100, cutmix_acc["pd_field"]["accuracy"] * 100]
x = np.arange(len(labels))
w = 0.35
ax.bar(x - w / 2, base_vals, w, label="Baseline (no CutMix)", color="#5B6472")
ax.bar(x + w / 2, cutmix_vals, w, label="CutMix training", color="#2F6F52")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Accuracy (%)")
ax.set_title("(a) Lab vs. field accuracy", fontsize=9)
ax.legend(fontsize=6.5, loc="upper right")
ax.grid(alpha=0.3, axis="y")
for i, v in enumerate(base_vals):
    ax.text(i - w / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=7)
for i, v in enumerate(cutmix_vals):
    ax.text(i + w / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=7)

# --- panel (b): Grad-CAM energy-in-box vs chance ---
ax2 = axes[1]
groups = ["Baseline", "CutMix"]
energy_vals = [baseline_gcam["all"]["mean_energy_in_box"] * 100, cutmix_gcam["all"]["mean_energy_in_box"] * 100]
chance_vals = [baseline_gcam["all"]["mean_chance"] * 100, cutmix_gcam["all"]["mean_chance"] * 100]
x2 = np.arange(len(groups))
ax2.bar(x2 - w / 2, chance_vals, w, label="Chance C(x)", color="#B0B0B0")
ax2.bar(x2 + w / 2, energy_vals, w, label="Attention E(x)", color="#B4622B")
ax2.set_xticks(x2)
ax2.set_xticklabels(groups, fontsize=8)
ax2.set_ylabel("Mean fraction (%)")
ax2.set_title("(b) Grad-CAM attention vs. chance\n(all 623 scorable field images)", fontsize=9)
ax2.set_ylim(0, 62)
ax2.legend(fontsize=6.5, loc="upper right")
ax2.grid(alpha=0.3, axis="y")
for i, v in enumerate(chance_vals):
    ax2.text(i - w / 2, v + 1.0, f"{v:.1f}", ha="center", fontsize=7)
for i, v in enumerate(energy_vals):
    ax2.text(i + w / 2, v + 1.0, f"{v:.1f}", ha="center", fontsize=7)

fig.tight_layout()
fig.savefig("figures/cutmix_comparison.png", dpi=220, bbox_inches="tight")
from PIL import Image
im = Image.open("figures/cutmix_comparison.png")
print("saved", im.size, "aspect", im.size[0] / im.size[1])
