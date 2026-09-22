"""
Four-way mechanism figure: does any intervention actually move the model's
attention back onto the leaf?

Each variant is scored by the identical metric -- the fraction of Grad-CAM
energy falling inside the annotated leaf box, E(x), minus the chance level
C(x) set by that box's own area, with a 10,000-resample bootstrap 95% CI on
the paired excess, read at the same named layer (block_13_expand_relu /
relu6, 6x6 at 96px input) in every model.

All four are scored on D_ft-test, the 584-image held-out field split, of
which 495 carry a usable leaf box. This matters: the adapted arms trained on
160 of the 744 field images, so scoring them over the full set would score
them partly on their own training data. The accuracy table already uses this
basis (Section 3.7); the attention numbers now use it too.

Plotting the excess rather than E(x) is the point: E(x) alone is not
comparable across images, because a photograph of a single large leaf has a
high chance level and a cluttered wide shot has a low one. Excess above zero
means "attends to the leaf more than its size alone would explain".

The figure separates correctly-classified from misclassified images because
that split is where the interventions actually differ.
"""
import json
import numpy as np
import figstyle as fs
import matplotlib.pyplot as plt

fs.apply()

VARIANTS = [
    ("Baseline", "baseline", fs.C_BEFORE),
    ("CutMix", "cutmix", fs.YELLOW),
    ("ImageNet-pretrained", "pretrained", fs.ORANGE),
    ("Pretrained + L2-SP", "stacked", fs.BLUE),
]

FT = json.load(open("gradcam_fttest_results.json"))["variants"]
data = [(label, col, FT[key]) for label, key, col in VARIANTS]
N_SCORED = FT["baseline"]["n_scored_fttest"]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(fs.FULL_W, 2.95),
                              gridspec_kw={"width_ratios": [1.25, 1.0]})

# ------------------------------------------- panel (a): excess by correctness
x = np.arange(len(data))
w = 0.36
for i, (grp, off, face, name) in enumerate([
        ("correct", -w / 2, None, "correctly classified"),
        ("misclassified", +w / 2, "hatch", "misclassified")]):
    vals = [d[grp]["mean_excess"] for _, _, d in data]
    los = [d[grp]["mean_excess"] - d[grp]["excess_ci95"][0] for _, _, d in data]
    his = [d[grp]["excess_ci95"][1] - d[grp]["mean_excess"] for _, _, d in data]
    cols = [c for _, c, _ in data]
    bars = ax.bar(x + off, vals, w * 0.92,
                  color=cols if face is None else "#ffffff",
                  edgecolor=cols, linewidth=1.1,
                  hatch=None if face is None else "////", zorder=3, label=name)
    ax.errorbar(x + off, vals, yerr=[los, his], fmt="none",
                ecolor=fs.INK_2, elinewidth=0.9, capsize=2.2, zorder=4)

ax.axhline(0, color=fs.INK_2, lw=0.9, zorder=2)
ax.set_xticks(x)
SHORT_X = ["Baseline", "CutMix", "Pretrained", "Pre + L2-SP"]
ax.set_xticklabels(SHORT_X, fontsize=fs.FS_TICK - 0.5)
ax.set_ylabel("Attention excess, $E(x)-C(x)$")
fs.panel_title(ax, "a", "Does attention move to the leaf?")
ax.set_ylim(-0.062, 0.090)
ax.legend(loc="lower left", fontsize=fs.FS_LEGEND, ncol=2)
fs.grid_y_only(ax)
fs.tick_labels(ax)


# --------------------------------- panel (b): how many images are even correct
n_cor = [d["correct"]["n"] for _, _, d in data]
n_mis = [d["misclassified"]["n"] for _, _, d in data]
ypos = np.arange(len(data))[::-1]
b1 = ax2.barh(ypos, n_cor, height=0.55, color=fs.AQUA, zorder=3, label="correct")
b2 = ax2.barh(ypos, n_mis, height=0.55, left=n_cor, color="#e6e4dc", zorder=3,
              label="misclassified")
for y, c in zip(ypos, n_cor):
    ax2.text(c / 2, y, str(c), va="center", ha="center",
             fontsize=fs.FS_ANNOT - 0.4, color="#ffffff", fontweight="bold")
ax2.set_yticks(ypos)
ax2.set_yticklabels([l.replace("\n", " ") for l, _, _ in data],
                    fontsize=fs.FS_TICK - 0.4)
ax2.set_xlabel(f"Scorable $D_{{ft\\text{{-}}test}}$ images (n = {N_SCORED})")
fs.panel_title(ax2, "b", "…and on how many images")
ax2.set_xlim(0, N_SCORED)
ax2.grid(True, axis="x")
ax2.grid(False, axis="y")
fs.tick_labels(ax2)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
           fontsize=fs.FS_LEGEND)

fs.finish(fig, "figures/attention_comparison.png")
