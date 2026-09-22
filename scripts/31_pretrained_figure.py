"""
Two-panel figure for the ImageNet-pretrained-backbone experiment
(29_pretrained_finetune.py + 30_pretrained_gradcam_bbox.py), added to
directly test the "tabula rasa confound" critique: is the from-scratch
model's 78.7%->12.1% lab-to-field collapse a domain-shift finding, or just
what small-dataset overfitting looks like without pretraining?

Panel (a): lab vs. field accuracy, from-scratch baseline vs. real
ImageNet-pretrained backbone (both evaluated on the same 744-image field set
and same held-out lab test set).
Panel (b): quantitative Grad-CAM energy-in-leaf-box vs. chance, baseline vs.
pretrained -- the mechanistic test of whether starting from real ImageNet
features redirects attention toward the leaf and away from background,
mirroring the same test already run for CutMix (Fig. 12).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

baseline_acc = json.load(open("results_tomato_fp32.json"))
pretrained_acc = json.load(open("results_tomato_pretrained_fp32.json"))
baseline_gcam = json.load(open("gradcam_bbox_energy_results.json"))
pretrained_gcam = json.load(open("gradcam_bbox_energy_pretrained_results.json"))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

# --- panel (a): lab vs field accuracy ---
ax = axes[0]
labels = ["Lab\n(PlantVillage)", "Field\n(PlantDoc, n=744)"]
base_vals = [baseline_acc["pv_test"]["accuracy"] * 100, baseline_acc["pd_field"]["accuracy"] * 100]
pre_vals = [pretrained_acc["pv_test"]["accuracy"] * 100, pretrained_acc["pd_field"]["accuracy"] * 100]
x = np.arange(len(labels))
w = 0.35
ax.bar(x - w / 2, base_vals, w, label="From-scratch (baseline)", color="#5B6472")
ax.bar(x + w / 2, pre_vals, w, label="ImageNet-pretrained", color="#1F6FA8")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Accuracy (%)")
ax.set_title("(a) Lab vs. field accuracy", fontsize=9)
ax.legend(fontsize=6.5, loc="upper right")
ax.grid(alpha=0.3, axis="y")
for i, v in enumerate(base_vals):
    ax.text(i - w / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=7)
for i, v in enumerate(pre_vals):
    ax.text(i + w / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=7)

# --- panel (b): Grad-CAM energy-in-box vs chance ---
ax2 = axes[1]
groups = ["From-scratch", "Pretrained"]
energy_vals = [baseline_gcam["all"]["mean_energy_in_box"] * 100, pretrained_gcam["all"]["mean_energy_in_box"] * 100]
chance_vals = [baseline_gcam["all"]["mean_chance"] * 100, pretrained_gcam["all"]["mean_chance"] * 100]
x2 = np.arange(len(groups))
ax2.bar(x2 - w / 2, chance_vals, w, label="Chance C(x)", color="#B0B0B0")
ax2.bar(x2 + w / 2, energy_vals, w, label="Attention E(x)", color="#B4622B")
ax2.set_xticks(x2)
ax2.set_xticklabels(groups, fontsize=8)
ax2.set_ylabel("Mean fraction (%)")
ax2.set_title(f"(b) Grad-CAM attention vs. chance\n(all {pretrained_gcam['n_scored']} scorable field images)", fontsize=9)
ymax = max(max(energy_vals), max(chance_vals)) * 1.3
ax2.set_ylim(0, ymax)
ax2.legend(fontsize=6.5, loc="upper right")
ax2.grid(alpha=0.3, axis="y")
for i, v in enumerate(chance_vals):
    ax2.text(i - w / 2, v + ymax * 0.02, f"{v:.1f}", ha="center", fontsize=7)
for i, v in enumerate(energy_vals):
    ax2.text(i + w / 2, v + ymax * 0.02, f"{v:.1f}", ha="center", fontsize=7)

fig.tight_layout()
fig.savefig("figures/pretrained_comparison.png", dpi=220, bbox_inches="tight")
from PIL import Image
im = Image.open("figures/pretrained_comparison.png")
print("saved", im.size, "aspect", im.size[0] / im.size[1])
