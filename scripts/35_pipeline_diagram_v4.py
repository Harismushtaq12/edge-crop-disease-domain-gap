"""
End-to-end pipeline diagram (Fig. 2), v4.

Replaces 22_architecture_diagram_v3.py, which had three problems: it was
visually noisy (six saturated fills, a legend floating free of the flow,
connectors crossing behind boxes), its boxes were sized by text length
rather than on a grid, and it was out of date -- it predated the CutMix,
ImageNet-pretrained and 224px-ablation arms, so it showed three of the
paper's mitigations instead of all of them.

Design: one calm surface. Nodes are light neutral cards carrying a 3px
accent stripe that encodes the stage, rather than six competing full-colour
fills; every node sits on a fixed column grid so the eye reads columns, not
box sizes; connectors are orthogonal, routed in the gutters between columns,
and always terminate at a card edge rather than passing behind one.
"""
import figstyle as fs
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

fs.apply()

# stage accent colours -- used only as a 3px stripe, never as a fill
ACC_DATA = fs.MUTED
ACC_PREP = fs.BLUE
ACC_TRAIN = fs.VIOLET
ACC_COMP = fs.ORANGE
ACC_EVAL = fs.AQUA
ACC_INTV = fs.YELLOW

CARD = "#f4f4f1"      # neutral card fill
CARD_EDGE = "#dedcd4"
STRIPE_W = 0.0055

fig, ax = plt.subplots(figsize=(fs.FULL_W, 3.30))
fs.blank_axes(ax)


def card(x, y, w, h, accent, title, lines, title_size=6.6, line_size=5.7,
         zorder=3):
    """A node: neutral card + left accent stripe + title + detail lines."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
        facecolor=CARD, edgecolor=CARD_EDGE, linewidth=0.7, zorder=zorder))
    # accent stripe, clipped to the card's rounded left edge
    ax.add_patch(Rectangle((x, y + 0.012), STRIPE_W, h - 0.024,
                           facecolor=accent, edgecolor="none", zorder=zorder + 1))
    tx = x + w / 2 + STRIPE_W / 2
    ax.text(tx, y + h - 0.048, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=fs.INK, zorder=zorder + 2)
    if lines:
        ax.text(tx, y + h - 0.048 - 0.062, "\n".join(lines), ha="center", va="top",
                fontsize=line_size, color=fs.INK_2, linespacing=1.5,
                zorder=zorder + 2)
    return (x, y, w, h)


def stage_header(x, w, label, accent):
    ax.text(x + w / 2, 0.965, label, ha="center", va="center",
            fontsize=6.6, fontweight="bold", color=accent)


def hline(x0, x1, y, color=None, lw=0.9, arrow_end=True, zorder=2):
    c = color or "#a9a7a0"
    style = "-|>" if arrow_end else "-"
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), transform=ax.transAxes,
                                 arrowstyle=style, mutation_scale=7,
                                 color=c, linewidth=lw, shrinkA=0, shrinkB=0,
                                 zorder=zorder, capstyle="round"))


def elbow(x0, y0, x1, y1, color=None, lw=0.9, arrow_end=True, first="h",
          zorder=2):
    """Orthogonal 2-segment connector routed through the column gutters."""
    c = color or "#a9a7a0"
    mid = (x1, y0) if first == "h" else (x0, y1)
    ax.add_patch(FancyArrowPatch((x0, y0), mid, transform=ax.transAxes,
                                 arrowstyle="-", color=c, linewidth=lw,
                                 shrinkA=0, shrinkB=0, zorder=zorder,
                                 capstyle="round"))
    style = "-|>" if arrow_end else "-"
    ax.add_patch(FancyArrowPatch(mid, (x1, y1), transform=ax.transAxes,
                                 arrowstyle=style, mutation_scale=7, color=c,
                                 linewidth=lw, shrinkA=0, shrinkB=0,
                                 zorder=zorder, capstyle="round"))


# ---------------------------------------------------------------- columns --
# fixed grid: 5 columns, equal gutters
COL_X = [0.006, 0.2093, 0.4126, 0.6159, 0.8192]
COL_W = 0.175
GUT = COL_X[1] - COL_X[0] - COL_W   # gutter width

stage_header(COL_X[0], COL_W, "1 · DATA", ACC_DATA)
stage_header(COL_X[1], COL_W, "2 · PREPARE", ACC_PREP)
stage_header(COL_X[2], COL_W, "3 · TRAIN", ACC_TRAIN)
stage_header(COL_X[3], COL_W, "4 · COMPRESS", ACC_COMP)
stage_header(COL_X[4], COL_W, "5 · EVALUATE", ACC_EVAL)

# ------------------------------------------------------------ top band: spine
Y_TOP = 0.735
H_SM = 0.175

card(COL_X[0], Y_TOP + 0.020, COL_W, H_SM, ACC_DATA, "PlantVillage",
     ["lab images, 21 classes", "capped 600/class"])
card(COL_X[0], Y_TOP - 0.215, COL_W, H_SM, ACC_DATA, "PlantDoc",
     ["field photos", "held out entirely"])

card(COL_X[1], Y_TOP - 0.145, COL_W, 0.325, ACC_PREP, "Preprocess & split",
     ["96×96, 70/15/15", "PlantVillage split", "all PlantDoc", "→ field set"])

card(COL_X[2], Y_TOP - 0.145, COL_W, 0.325, ACC_TRAIN, "Train from scratch",
     ["MobileNetV2", "MobileNetV3-Small", "EfficientNetB0", "shared recipe"])

card(COL_X[3], Y_TOP - 0.145, COL_W, 0.325, ACC_COMP, "INT8 quantize",
     ["TFLite post-training", "200-image calib.", "FP32 export kept", "for comparison"])

card(COL_X[4], Y_TOP + 0.048, COL_W, 0.148, ACC_EVAL, "Lab test",
     ["PlantVillage", "held-out split"])
card(COL_X[4], Y_TOP - 0.148, COL_W, 0.148, ACC_EVAL, "Field test",
     ["PlantDoc,", "never seen in training"])

# spine connectors
y_mid = Y_TOP + 0.0175
merge_x = COL_X[0] + COL_W + GUT * 0.45
elbow(COL_X[0] + COL_W, Y_TOP + 0.108, merge_x, y_mid, arrow_end=False)
elbow(COL_X[0] + COL_W, Y_TOP - 0.128, merge_x, y_mid, arrow_end=False)
hline(merge_x, COL_X[1], y_mid)
hline(COL_X[1] + COL_W, COL_X[2], y_mid)
hline(COL_X[2] + COL_W, COL_X[3], y_mid)
split_x = COL_X[3] + COL_W + GUT * 0.55
hline(COL_X[3] + COL_W, split_x, y_mid, arrow_end=False)
elbow(split_x, y_mid, COL_X[4], Y_TOP + 0.122, first="v")
elbow(split_x, y_mid, COL_X[4], Y_TOP - 0.074, first="v")

# --------------------------------------------------- right: the headline metric
# The paper's central quantity gets its own emphasised card, set apart from
# the neutral pipeline cards so the eye lands on it last and stays there.
DY0, DH = 0.400, 0.125
ax.add_patch(FancyBboxPatch(
    (COL_X[4], DY0), COL_W, DH, boxstyle="round,pad=0,rounding_size=0.012",
    facecolor="#eaf2fd", edgecolor=fs.BLUE, linewidth=0.9, zorder=4))
ax.text(COL_X[4] + COL_W / 2, DY0 + DH * 0.60,
        "Δ$_{domain}$ = Acc$_{lab}$ − Acc$_{field}$",
        ha="center", va="center", fontsize=6.5, fontweight="bold",
        color=fs.INK, zorder=6)
ax.text(COL_X[4] + COL_W / 2, DY0 + DH * 0.22,
        "what every arm below targets", ha="center", va="center",
        fontsize=5.7, color=fs.INK_2, zorder=6)
elbow(COL_X[4] + COL_W * 0.5, Y_TOP - 0.148, COL_X[4] + COL_W * 0.5, DY0 + DH,
      first="v", arrow_end=True)

# ------------------------------------------------ bottom band: interventions
Y_INT = 0.075
H_INT = 0.215
lane_y0, lane_y1 = Y_INT - 0.028, Y_INT + H_INT + 0.078
ax.add_patch(FancyBboxPatch(
    (0.006, lane_y0), 0.988, lane_y1 - lane_y0,
    boxstyle="round,pad=0,rounding_size=0.012",
    facecolor="#fbfaf7", edgecolor="#e6e4dc", linewidth=0.7, zorder=1))
ax.text(0.016, lane_y1 - 0.032,
        "FIVE INTERVENTIONS TESTED AGAINST Δ$_{domain}$ — none closes it",
        ha="left", va="center", fontsize=6.4, fontweight="bold",
        color="#9a6a00", zorder=2)

IW = 0.1848
IX = [0.014, 0.2108, 0.4076, 0.6044, 0.8012]
intv = [
    ("Few-shot field FT", ["full / head-only / L2-SP", "15 field imgs/class"]),
    ("CutMix", ["training-time aug.", "zero field data"]),
    ("ImageNet-pretrained", ["ported 96px ckpt", "zero field data"]),
    ("Pretrained + field FT", ["stacked combination", "of the two best arms"]),
    ("224px resolution", ["224-native ckpt", "resolution ablation"]),
]
for x, (t, ls) in zip(IX, intv):
    card(x, Y_INT, IW, H_INT, ACC_INTV, t, ls, title_size=7.0, line_size=6.1)

# interventions feed back into TRAIN, and are re-measured by EVALUATE
feed_y = Y_INT + H_INT + 0.006
elbow(COL_X[2] + COL_W * 0.5, feed_y + 0.052, COL_X[2] + COL_W * 0.5,
      Y_TOP - 0.145, first="v", color="#c2bfb4", lw=0.9)
ax.text(COL_X[2] + COL_W * 0.5 + 0.006, (feed_y + 0.052 + Y_TOP - 0.145) / 2,
        "re-train / re-init", ha="left", va="center", fontsize=5.9,
        color=fs.MUTED, style="italic", zorder=5)

# ------------------------------------------------------------- analyses note
ax.text(0.984, lane_y1 - 0.032,
        "each re-scored on both sets + Grad-CAM leaf-box attention",
        ha="right", va="center", fontsize=5.9, color=fs.MUTED, zorder=2)

fs.finish(fig, "figures/pipeline_v4.png", tight=False)
