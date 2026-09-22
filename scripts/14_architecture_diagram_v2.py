"""
v2 of the architecture/pipeline diagram (paper Figure 1). Complete redesign
from v1: a single left-to-right flow (no crossing arrows), a restrained
5-color categorical palette, consistent flat-design boxes (no drop shadows,
uniform stroke width), and a compact landscape aspect ratio so it reads as
one clean hero figure instead of a tall, busy multi-branch diagram.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(13.2, 5.8))
ax.set_xlim(0, 150)
ax.set_ylim(0, 66)
ax.axis("off")

# ---------------------------------------------------------------- palette --
# One restrained, colorblind-considerate categorical palette: five hues,
# each used consistently for one pipeline concern. No pure red/green clash,
# no drop shadows, uniform border weight -- flat, modern, "systems diagram"
# style rather than a decorative one.
INK = "#28313E"          # universal border / arrow color
DATA_FILL, DATA_TXT = "#E4EAF3", "#1C2B3A"          # pale slate-blue
PREP_FILL, PREP_TXT = "#D9E6F2", "#1C2B3A"          # slightly deeper slate-blue
MODEL_FILL, MODEL_TXT = "#1E3A5F", "#FFFFFF"        # deep navy
QUANT_FILL, QUANT_TXT = "#B4622B", "#FFFFFF"        # warm terracotta
EVAL_FILL, EVAL_TXT = "#2F6F52", "#FFFFFF"          # forest green
MIT_FILL, MIT_TXT = "#6B4E71", "#FFFFFF"            # muted plum
NOTE_FILL, NOTE_TXT = "#EDEDED", "#28313E"          # neutral gray

def box(x, y, w, h, text, fc, tc, fontsize=8.6, weight="normal", ls="-", lw=1.15, ec=INK):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.30,rounding_size=1.3",
                        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=tc, weight=weight, zorder=3, linespacing=1.32)
    return (x, y, w, h)

def arrow(p1, p2, color=INK, lw=1.3, connectionstyle="arc3,rad=0.0", ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11, linewidth=lw,
                                  color=color, connectionstyle=connectionstyle, zorder=1, linestyle=ls))

def right(bx, f=0.5): return (bx[0] + bx[2], bx[1] + bx[3] * f)
def left(bx, f=0.5): return (bx[0], bx[1] + bx[3] * f)
def top(bx, f=0.5): return (bx[0] + bx[2] * f, bx[1] + bx[3])
def bottom(bx, f=0.5): return (bx[0] + bx[2] * f, bx[1])

# ============================================================== legend ====
# Drawn first, in its own reserved strip at the very top, well clear of
# every box below it.
legend_items = [
    ("Data", DATA_FILL, DATA_TXT), ("Preprocessing / split", PREP_FILL, PREP_TXT),
    ("Model training", MODEL_FILL, MODEL_TXT), ("Quantization", QUANT_FILL, QUANT_TXT),
    ("Evaluation", EVAL_FILL, EVAL_TXT), ("Mitigation", MIT_FILL, MIT_TXT),
]
lx = 2
ly = 64.3
for label, fc, tc in legend_items:
    ax.add_patch(FancyBboxPatch((lx, ly - 1.1), 2.2, 2.2, boxstyle="round,pad=0.08,rounding_size=0.35",
                                 facecolor=fc, edgecolor=INK, linewidth=0.9))
    ax.text(lx + 2.9, ly, label, fontsize=7.8, va="center", color=INK)
    lx += len(label) * 0.62 + 6.6

# ============================================================ main row ====
# A single left-to-right pass: data -> preprocess/split -> train -> FP32 ->
# INT8 -> lab/field eval. Everything reads in one direction, kept well
# below the legend strip above.
pv = box(2, 46, 21, 16, "PlantVillage\n(lab images)\nHughes & Salathe 2015",
         DATA_FILL, DATA_TXT, fontsize=7.9)
pd_ = box(2, 24, 21, 16, "PlantDoc\n(field images)\nSingh et al. 2020",
          DATA_FILL, DATA_TXT, fontsize=7.9)

prep = box(28, 22, 21, 32, "Preprocess\n+ split\n\n70/15/15\ntrain/val/\nlab-test;\nall PlantDoc\n→ field set\n(§IV-A)",
           PREP_FILL, PREP_TXT, fontsize=7.6)
arrow(right(pv), left(prep, 0.78), connectionstyle="arc3,rad=-0.12")
arrow(right(pd_), left(prep, 0.22), connectionstyle="arc3,rad=0.12")

train = box(54, 16, 25, 42,
            "Train 3 backbones\nfrom scratch\n\nMobileNetV2\nMobileNetV3-Small\nEfficientNetB0\n\nshared recipe\n(§IV-B, IV-E)",
            MODEL_FILL, MODEL_TXT, fontsize=7.9)
arrow(right(prep), left(train), connectionstyle="arc3,rad=0.0")

fp32 = box(84, 42, 20, 16, "FP32 model\n(per arch. ×\nexperiment)", MODEL_FILL, MODEL_TXT, fontsize=7.9)
arrow(right(train), left(fp32), connectionstyle="arc3,rad=0.0")

int8 = box(84, 22, 20, 16, "INT8 quantize\n200-image\ncalibration\n(§IV-C)", QUANT_FILL, QUANT_TXT, fontsize=7.9)
arrow(bottom(fp32), top(int8), connectionstyle="arc3,rad=0.0")

labeval = box(109, 42, 22, 16, "Lab eval\n(pv_test)", EVAL_FILL, EVAL_TXT, fontsize=8.0)
fieldeval = box(109, 22, 22, 16, "Field eval\n(PlantDoc)\nFig. 4 confusion\nmatrix", EVAL_FILL, EVAL_TXT, fontsize=7.5)
arrow(right(int8), (labeval[0], labeval[1] + labeval[3] * 0.15), connectionstyle="arc3,rad=0.22")
arrow(right(int8), left(fieldeval), connectionstyle="arc3,rad=0.0")
arrow(right(fp32), (labeval[0], labeval[1] + labeval[3] * 0.85), connectionstyle="arc3,rad=-0.06")

# a light dashed tap from FP32 down to field-eval, showing FP32 is also
# evaluated on both sets (not only the INT8 export) -- kept understated.
arrow(bottom(fp32), (fieldeval[0] + 2, fieldeval[1] + fieldeval[3] + 1),
      connectionstyle="arc3,rad=-0.28", ls="--", color="#8A93A0", lw=0.9)

# ============================================================ delta note ==
delta = box(134, 30, 14, 16, "Δ_domain\n=\nAcc_lab −\nAcc_field\n(Eq. 5)", NOTE_FILL, NOTE_TXT, fontsize=7.4)
arrow(right(labeval), (delta[0], delta[1] + delta[3] * 0.75), connectionstyle="arc3,rad=-0.15")
arrow(right(fieldeval), (delta[0], delta[1] + delta[3] * 0.25), connectionstyle="arc3,rad=0.15")

# ============================================================ mitigation ==
# A clearly separated bottom strip of its own, aligned directly under the
# stages it touches (train / FP32 / INT8), connected by one short dashed
# branch -- not woven through the main flow.
mit = box(54, 2, 50, 12,
          "Field fine-tuning mitigation (tomato, MobileNetV2)\n"
          "full vs. head-only — 15/cls train, 5/cls val,\n"
          "remainder test → re-eval on field + lab (§IV-F, V-E)",
          MIT_FILL, MIT_TXT, fontsize=7.3)
arrow(top(mit, 0.75), (fp32[0] + 3, fp32[1]),
      connectionstyle="arc3,rad=0.15", ls="--", color=MIT_FILL, lw=1.2)

fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
fig.savefig("figures/architecture_pipeline.png", dpi=230, bbox_inches="tight")
from PIL import Image
print("saved", Image.open("figures/architecture_pipeline.png").size)
