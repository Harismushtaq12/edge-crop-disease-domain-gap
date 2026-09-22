"""
Bar chart for the finer per-block INT8 sensitivity sweep
(20_layer_sensitivity_blocks.py): lab-test accuracy vs. which layer group
was fake-quantized, isolating the stem as the true cause of the
MobileNetV3-Small collapse (Section V-D).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("layer_sensitivity_blocks_results.json"))
baseline = d["baseline_fp32_acc"]
order = ["Stem", "Blocks 1-3", "Blocks 4-7", "Blocks 8-11", "SE units", "Head"]
accs = [d["groups"][k]["acc"] for k in order]

STEM_COL = "#B3261E"
OTHER_COL = "#2F6F52"
colors = [STEM_COL if k == "Stem" else OTHER_COL for k in order]

fig, ax = plt.subplots(figsize=(3.35, 2.35))
x = np.arange(len(order))
bars = ax.bar(x, accs, color=colors, width=0.62, zorder=3)
ax.axhline(baseline, color="#28313E", linestyle="--", linewidth=1.1, zorder=2,
           label=f"FP32 baseline ({baseline:.3f})")
for xi, a in zip(x, accs):
    ax.text(xi, a + 0.018, f"{a:.2f}", ha="center", va="bottom", fontsize=6.6)
ax.set_xticks(x)
ax.set_xticklabels(order, fontsize=6.6, rotation=28, ha="right")
ax.set_ylabel("Lab-test accuracy", fontsize=7.6)
ax.set_ylim(0, max(baseline, max(accs)) * 1.28)
ax.tick_params(axis="y", labelsize=7)
ax.legend(fontsize=6.0, frameon=False, loc="upper right")
ax.set_title("Only the stem drives the INT8 collapse", fontsize=7.9)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.grid(axis="y", linewidth=0.4, color="#DADFE5", zorder=0)

fig.tight_layout(pad=0.4)
fig.savefig("figures/layer_sensitivity_blocks.png", dpi=300, bbox_inches="tight")
print("Saved figures/layer_sensitivity_blocks.png")
