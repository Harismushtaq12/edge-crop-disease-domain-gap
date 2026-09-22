"""
FARM framework diagram (Fig. 1), v2.

Replaces 23_farm_diagram.py. The old version was a 2x2 of saturated colour
blocks with a dashed feedback arrow that started in dead space and crossed
two of the boxes, and -- more importantly -- it was purely declarative: it
asserted four stages without showing that any of them was grounded in a
measurement.

This version is built around the revision's decision to keep FARM only as a
data-anchored procedure: every stage carries the gate it imposes AND the
specific measured result in this paper that motivates that gate. Stages
whose evidence comes from experiments here are marked as measured; Rollout
and Maintenance, which this paper does not execute over a season, are
explicitly marked as design guidance so the figure itself makes the
framework's limits visible rather than overclaiming.

Layout: a single left-to-right row (easier to read in print than a 2x2
cycle) with a return arc underneath carrying the drift/failure feedback,
which is what makes it a cycle rather than a pipeline.
"""
import figstyle as fs
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle

fs.apply()

CARD = "#f4f4f1"
CARD_EDGE = "#dedcd4"
MEASURED = fs.BLUE        # stage backed by an experiment in this paper
GUIDANCE = "#9b9a94"       # stage stated as design guidance, not validated here

FIG_W, FIG_H = fs.FULL_W, 2.16
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fs.blank_axes(ax)

STAGES = [
    ("F", "Feasibility", MEASURED,
     "Gate: an independently collected\nfield-validation set, before any pilot",
     "78.7–86.8% lab → 9.1–12.1% field;\na 97% pretrained model still fails"),
    ("A", "Architecture", MEASURED,
     "Gate: re-measure compression\ncost per backbone, never assume it",
     "all 3 backbones 11.6–14.4% field;\nINT8 costs MobileNetV3-S 44 pp"),
    ("R", "Rollout", GUIDANCE,
     "Gate: staged human verification,\nscaled to the measured gap",
     "the best arm still leaves a 45 pp gap\n— pilot accordingly"),
    ("M", "Maintenance", GUIDANCE,
     "Gate: per-class field tracking,\nnot one aggregate number",
     "per-class field accuracy spans\n0–38.6% — an average hides that"),
]

X0, GAP = 0.006, 0.026
W = (0.988 - 3 * GAP) / 4
Y, H = 0.300, 0.583

for i, (letter, name, accent, gate, evidence) in enumerate(STAGES):
    x = X0 + i * (W + GAP)
    ax.add_patch(FancyBboxPatch(
        (x, Y), W, H, boxstyle="round,pad=0,rounding_size=0.020",
        facecolor=CARD, edgecolor=CARD_EDGE, linewidth=0.8, zorder=3))
    # top accent rule -- encodes measured vs. guidance
    ax.add_patch(Rectangle((x + 0.010, Y + H - 0.016), W - 0.020, 0.012,
                           facecolor=accent, edgecolor="none", zorder=4))

    # letter badge
    cx, cy = x + 0.026, Y + H - 0.105
    d_in = 0.165  # target visual diameter, inches
    ax.add_patch(Ellipse((cx, cy), width=d_in / FIG_W, height=d_in / FIG_H,
                         facecolor=accent, edgecolor="none", zorder=5))
    ax.text(cx, cy, letter, ha="center", va="center", fontsize=8.5,
            fontweight="bold", color="#ffffff", zorder=6)
    ax.text(cx + 0.030, cy, name, ha="left", va="center", fontsize=7.9,
            fontweight="bold", color=fs.INK, zorder=6)

    ax.text(x + 0.016, Y + H - 0.185, gate, ha="left", va="top",
            fontsize=5.9, color=fs.INK, linespacing=1.45, zorder=6)

    # evidence block, separated by a hairline
    ax.plot([x + 0.016, x + W - 0.016], [Y + 0.150, Y + 0.150],
            color="#e1e0d9", lw=0.7, zorder=5)
    tag = "MEASURED HERE" if accent is MEASURED else "NOT VALIDATED HERE"
    ax.text(x + 0.016, Y + 0.120, tag, ha="left", va="center", fontsize=5.4,
            fontweight="bold", color=accent, zorder=6)
    ax.text(x + 0.016, Y + 0.098, evidence, ha="left", va="top",
            fontsize=5.6, color=fs.INK_2, linespacing=1.4, zorder=6)

    # forward arrow
    if i < len(STAGES) - 1:
        ax.add_patch(FancyArrowPatch(
            (x + W + 0.004, Y + H / 2), (x + W + GAP - 0.004, Y + H / 2),
            transform=ax.transData, arrowstyle="-|>", mutation_scale=8,
            color="#a9a7a0", linewidth=1.0, shrinkA=0, shrinkB=0, zorder=4))

# ------------------------------------------------- feedback arc M -> F
last_x = X0 + 3 * (W + GAP)
FB_Y = 0.105          # the return lane, clear of every card
DASH = (0, (4, 2.4))
seg = dict(transform=ax.transData, color=fs.ORANGE, linewidth=1.1,
           linestyle=DASH, shrinkA=0, shrinkB=0, zorder=4, capstyle="round")
mx, fx = last_x + W / 2, X0 + W / 2
ax.add_patch(FancyArrowPatch((mx, Y), (mx, FB_Y), arrowstyle="-", **seg))
ax.add_patch(FancyArrowPatch((mx, FB_Y), (fx, FB_Y), arrowstyle="-", **seg))
ax.add_patch(FancyArrowPatch((fx, FB_Y), (fx, Y), arrowstyle="-|>",
                             mutation_scale=9, **seg))
ax.text(0.5, FB_Y + 0.052,
        "drift / failure detected → re-audit Feasibility, don't absorb it silently",
        ha="center", va="center", fontsize=6.3, color=fs.ORANGE,
        fontweight="bold", zorder=5)

# ------------------------------------------------------------------ legend
ax.add_patch(Rectangle((X0, 0.962), 0.020, 0.0105, facecolor=MEASURED,
                       edgecolor="none", zorder=5))
ax.text(X0 + 0.028, 0.967, "gate backed by an experiment in this paper",
        ha="left", va="center", fontsize=6.0, color=fs.INK_2, zorder=5)
ax.add_patch(Rectangle((0.430, 0.962), 0.020, 0.0105, facecolor=GUIDANCE,
                       edgecolor="none", zorder=5))
ax.text(0.458, 0.967, "design guidance the results argue for, not separately validated here",
        ha="left", va="center", fontsize=6.0, color=fs.INK_2, zorder=5)

fs.finish(fig, "figures/farm_v2.png", tight=False)
