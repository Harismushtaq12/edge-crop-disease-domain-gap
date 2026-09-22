"""
The paper's central argument as one figure: every tomato variant placed in
the same lab-accuracy vs field-accuracy plane, on the one field split that
is held out from all of them (D_ft-test, 584 images).

Panel (a) shows why the "nothing works" reading was too strong and the
"pretraining fixes it" reading was too weak. The from-scratch family traces
a trade-off band -- each mitigation buys a little field accuracy by giving
up a lot of lab accuracy, sliding left along the band without rising much.
The ImageNet-pretrained family sits on a strictly higher band: at
essentially the baseline's own lab accuracy, the stacked pretrained + L2-SP
variant reaches 2.5x the baseline's field accuracy. That is a dominance
claim, and it is the thing a practitioner actually wants to know.

Panel (b) prices each intervention: field-accuracy gained per point of lab
accuracy surrendered, relative to the shared from-scratch baseline.

Inputs: unified_scoreboard.json (37_unified_scoreboard.py).
"""
import json
import figstyle as fs
import matplotlib.pyplot as plt
import numpy as np

fs.apply()

sb = json.load(open("unified_scoreboard.json"))
V = {v["label"]: v for v in sb["variants"]}
CHANCE = 12.5  # 8 tomato classes

SCRATCH_FAM = ["From-scratch baseline", "+ full fine-tune", "+ head-only fine-tune",
               "+ L2-SP fine-tune", "+ CutMix"]
PRE_FAM = ["ImageNet-pretrained", "Pretrained + head-only", "Pretrained + L2-SP"]

SHORT = {
    "From-scratch baseline": "baseline",
    "+ full fine-tune": "full FT",
    "+ head-only fine-tune": "head-only FT",
    "+ L2-SP fine-tune": "L2-SP",
    "+ CutMix": "CutMix",
    "ImageNet-pretrained": "pretrained",
    "Pretrained + head-only": "pre + head-only",
    "Pretrained + L2-SP": "pre + L2-SP",
}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(fs.FULL_W, 3.30),
                              gridspec_kw={"width_ratios": [1.15, 1.0]})

# ----------------------------------------------------------------- panel (a)
def xy(label):
    v = V[label]
    return v["lab"] * 100, v["field_fttest"] * 100

ax.axhline(CHANCE, color=fs.MUTED, lw=0.9, ls=(0, (4, 2.5)), zorder=1)
ax.text(29.0, CHANCE - 0.5, "chance (8 classes)", fontsize=fs.FS_ANNOT - 0.3,
        color=fs.MUTED, va="top", ha="left")

# family bands, drawn first so markers sit on top
sx = [xy(l)[0] for l in SCRATCH_FAM]
sy = [xy(l)[1] for l in SCRATCH_FAM]
px = [xy(l)[0] for l in PRE_FAM]
py = [xy(l)[1] for l in PRE_FAM]
# each mitigation is joined to the base model it derives from, so the
# lines encode lineage rather than an imaginary trajectory through the plane
for child in ["+ full fine-tune", "+ head-only fine-tune",
              "+ L2-SP fine-tune", "+ CutMix"]:
    x0, y0 = xy("From-scratch baseline")
    x1, y1 = xy(child)
    ax.plot([x0, x1], [y0, y1], color=fs.C_BEFORE, lw=0.9, alpha=0.5, zorder=2)
for child in ["Pretrained + head-only", "Pretrained + L2-SP"]:
    x0, y0 = xy("ImageNet-pretrained")
    x1, y1 = xy(child)
    ax.plot([x0, x1], [y0, y1], color=fs.ORANGE, lw=0.9, alpha=0.5, zorder=2)

ax.scatter(sx, sy, s=34, facecolor=fs.C_BEFORE, edgecolor="#ffffff",
           linewidth=1.0, zorder=4, label="trained from scratch")
ax.scatter(px, py, s=34, facecolor=fs.ORANGE, edgecolor="#ffffff",
           linewidth=1.0, zorder=4, label="ImageNet-pretrained backbone")

# the two points the argument turns on
for lbl, dx, dy, ha in [
    ("From-scratch baseline", 0, -3.0, "center"),
    ("+ full fine-tune", 0, 1.6, "center"),
    ("+ head-only fine-tune", 0, 1.6, "center"),
    ("+ L2-SP fine-tune", 1.5, 1.4, "left"),
    ("+ CutMix", 0, -3.0, "center"),
    ("ImageNet-pretrained", 0, -3.4, "center"),
    ("Pretrained + head-only", 0.5, 1.6, "center"),
    ("Pretrained + L2-SP", -1.5, 1.2, "right"),
]:
    x, y = xy(lbl)
    ax.annotate(SHORT[lbl], (x, y), textcoords="offset points",
                xytext=(dx * 3.2, dy * 3.2), ha=ha, fontsize=fs.FS_ANNOT - 0.4,
                color=fs.INK, zorder=6)

# highlight the dominance comparison
bx, by = xy("From-scratch baseline")
tx, ty = xy("Pretrained + L2-SP")
ax.annotate("", xy=(tx, ty), xytext=(bx, by),
            arrowprops=dict(arrowstyle="-|>", color=fs.BLUE, lw=1.3,
                            shrinkA=6, shrinkB=6,
                            connectionstyle="arc3,rad=-0.25"), zorder=5)

ax.set_xlabel("Lab accuracy, PlantVillage held-out (%)")
ax.set_ylabel("Field accuracy, $D_{ft\\text{-}test}$ (%)")
fs.panel_title(ax, "a", "The trade-off plane")
ax.set_xlim(28, 112)
ax.set_ylim(8, 41)
ax.legend(loc="upper left", fontsize=fs.FS_LEGEND)
ax.grid(True)

# ----------------------------------------------------------------- panel (b)
base_lab, base_field = xy("From-scratch baseline")
order = ["+ full fine-tune", "+ head-only fine-tune", "+ L2-SP fine-tune",
         "+ CutMix", "ImageNet-pretrained", "Pretrained + head-only",
         "Pretrained + L2-SP"]
gains = [xy(l)[1] - base_field for l in order]
costs = [xy(l)[0] - base_lab for l in order]
ypos = np.arange(len(order))[::-1]

ax2.axvline(0, color=fs.AXIS, lw=0.8, zorder=2)
h_gain = ax2.barh(ypos + 0.19, gains, height=0.34, color=fs.BLUE, zorder=3)
h_cost = ax2.barh(ypos - 0.19, costs, height=0.34, color=fs.ORANGE, zorder=3)

for y, g in zip(ypos + 0.19, gains):
    ax2.text(g + 0.7, y, f"+{g:.1f}", va="center", ha="left",
             fontsize=fs.FS_ANNOT - 0.6, color=fs.INK)
for y, c in zip(ypos - 0.19, costs):
    ax2.text(c + (0.9 if c >= 0 else -1.6), y, f"{c:+.1f}", va="center",
             ha="left" if c >= 0 else "right",
             fontsize=fs.FS_ANNOT - 0.6, color=fs.INK)

ax2.set_yticks(ypos)
ax2.set_yticklabels([SHORT[l] for l in order], fontsize=fs.FS_TICK - 0.3)
ax2.set_xlabel("Change vs. from-scratch baseline (pp)")
fs.panel_title(ax2, "b", "What each costs")
ax2.set_xlim(-66, 32)
ax2.grid(True, axis="x")
ax2.grid(False, axis="y")
fs.tick_labels(ax2)

from matplotlib.patches import Patch
ax2.legend(handles=[Patch(facecolor=fs.BLUE, label="field accuracy change"),
                    Patch(facecolor=fs.ORANGE, label="lab accuracy change")],
           loc="upper left", bbox_to_anchor=(0.0, -0.20), ncol=2,
           fontsize=fs.FS_LEGEND)

fs.finish(fig, "figures/tradeoff_frontier.png")
