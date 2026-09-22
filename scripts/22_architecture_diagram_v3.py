"""
v3 of the architecture/pipeline diagram (paper Fig. 1). Complete redesign
from v2, for two reasons: (1) v2 was being placed in the LaTeX/DOCX source
at a tiny fraction of the available width, making it illegible in the
compiled paper regardless of its own quality, and (2) a genuine polish
pass -- numbered stage labels (like a real systems-diagram convention),
a flatter aspect ratio so it stays compact even at near-full page width,
tighter and more consistent box sizing, and a condensed single-line
legend, so it reads as a clean, deliberate hero figure rather than a
cluttered flowchart of default rectangles.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(15.4, 4.55))
ax.set_xlim(0, 163)
ax.set_ylim(0, 34.5)
ax.axis("off")

INK = "#242C38"
DATA_FILL, DATA_TXT = "#E4EAF3", "#1C2B3A"
PREP_FILL, PREP_TXT = "#CBDCEE", "#1C2B3A"
MODEL_FILL, MODEL_TXT = "#1E3A5F", "#FFFFFF"
QUANT_FILL, QUANT_TXT = "#B4622B", "#FFFFFF"
EVAL_FILL, EVAL_TXT = "#2F6F52", "#FFFFFF"
MIT_FILL, MIT_TXT = "#6B4E71", "#FFFFFF"
NOTE_FILL, NOTE_TXT = "#EEEFF1", "#242C38"
STAGE_COL = "#5B6472"

def box(x, y, w, h, text, fc, tc, fontsize=9.6, weight="normal", lw=1.25, ec=INK,
        stage=None, stage_col=STAGE_COL):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.32,rounding_size=1.35",
                        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=tc, weight=weight, zorder=3, linespacing=1.38)
    if stage:
        ax.text(x + w / 2, y + h + 1.05, stage, ha="center", va="bottom", fontsize=7.7,
                 color=stage_col, weight="bold", zorder=3,
                 family="DejaVu Sans")
    return (x, y, w, h)

def arrow(p1, p2, color=INK, lw=1.5, connectionstyle="arc3,rad=0.0", ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13, linewidth=lw,
                                  color=color, connectionstyle=connectionstyle, zorder=1, linestyle=ls))

def right(bx, f=0.5): return (bx[0] + bx[2], bx[1] + bx[3] * f)
def left(bx, f=0.5): return (bx[0], bx[1] + bx[3] * f)
def top(bx, f=0.5): return (bx[0] + bx[2] * f, bx[1] + bx[3])
def bottom(bx, f=0.5): return (bx[0] + bx[2] * f, bx[1])

# ============================================================== legend ====
legend_items = [
    ("Data", DATA_FILL), ("Preprocessing / split", PREP_FILL),
    ("Model training", MODEL_FILL), ("Quantization", QUANT_FILL),
    ("Evaluation", EVAL_FILL), ("Mitigation", MIT_FILL),
]
total_w = sum(len(l) * 0.585 + 7.6 for l, _ in legend_items)
lx = (163 - total_w) / 2
ly = 32.6
for label, fc in legend_items:
    ax.add_patch(FancyBboxPatch((lx, ly - 0.95), 1.9, 1.9, boxstyle="round,pad=0.06,rounding_size=0.3",
                                 facecolor=fc, edgecolor=INK, linewidth=0.9, zorder=3))
    ax.text(lx + 2.55, ly, label, fontsize=8.4, va="center", color=INK, zorder=3)
    lx += len(label) * 0.585 + 7.6

# ============================================================ main row ====
ROW_Y, ROW_H = 8.2, 18.0

pv = box(2, ROW_Y + 9.8, 19.5, 8.2, "PlantVillage\nlab images (54k)\nHughes & Salathe 2015",
         DATA_FILL, DATA_TXT, fontsize=8.0, stage="1 · DATA")
pd_ = box(2, ROW_Y, 19.5, 8.2, "PlantDoc\nfield images (2.6k)\nSingh et al. 2020",
          DATA_FILL, DATA_TXT, fontsize=8.0)
arrow((21.5, ROW_Y + 14), (21.5, ROW_Y + 13.9))  # no-op placeholder to keep structure symmetric

prep = box(26, ROW_Y, 20, ROW_H, "Preprocess\n+ split\n\n70/15/15\ntrain / val /\nlab-test;\nall PlantDoc\n→ field set\n(§IV-A)",
           PREP_FILL, PREP_TXT, fontsize=7.9, stage="2 · PREPARE")
arrow(right(pv), left(prep, 0.78), connectionstyle="arc3,rad=-0.15")
arrow(right(pd_), left(prep, 0.22), connectionstyle="arc3,rad=0.15")

train = box(51, ROW_Y, 25, ROW_H,
            "Train 3 backbones\nfrom scratch\n\nMobileNetV2\nMobileNetV3-Small\nEfficientNetB0\n\nshared recipe (§IV-B, IV-E)",
            MODEL_FILL, MODEL_TXT, fontsize=8.4, stage="3 · TRAIN")
arrow(right(prep), left(train))

fp32 = box(81, ROW_Y + 9.8, 20, 8.2, "FP32 model\n(per architecture\n× experiment)",
           MODEL_FILL, MODEL_TXT, fontsize=8.0, stage="4 · QUANTIZE")
arrow(right(train), left(fp32, 0.78), connectionstyle="arc3,rad=-0.12")
int8 = box(81, ROW_Y, 20, 8.2, "INT8 quantize\n200-image\ncalibration (§IV-C)",
           QUANT_FILL, QUANT_TXT, fontsize=8.0)
arrow(bottom(fp32, 0.3), top(int8, 0.3), lw=1.3)
arrow(right(train), left(int8, 0.22), connectionstyle="arc3,rad=0.12")

labeval = box(106, ROW_Y + 9.8, 22, 8.2, "Lab eval\n(pv-test)",
              EVAL_FILL, EVAL_TXT, fontsize=8.6, stage="5 · EVALUATE")
fieldeval = box(106, ROW_Y, 22, 8.2, "Field eval\n(PlantDoc)\nFig. 4 conf. matrix",
                EVAL_FILL, EVAL_TXT, fontsize=7.8)
arrow(right(fp32), left(labeval, 0.78), connectionstyle="arc3,rad=-0.10")
arrow(right(int8), left(fieldeval, 0.22), connectionstyle="arc3,rad=0.10")
arrow(right(int8), (labeval[0], labeval[1] + labeval[3] * 0.15), connectionstyle="arc3,rad=0.32", lw=1.1)
arrow(bottom(fp32), (fieldeval[0] + 2, fieldeval[1] + fieldeval[3] + 0.9),
      connectionstyle="arc3,rad=-0.3", ls="--", color="#8A93A0", lw=1.0)

delta = box(133, ROW_Y + 3.5, 15, 11, "Δdomain\n=\nAcc_lab − Acc_field\n(Eq. 5)",
            NOTE_FILL, NOTE_TXT, fontsize=8.2)
arrow(right(labeval), (delta[0], delta[1] + delta[3] * 0.78), connectionstyle="arc3,rad=-0.18")
arrow(right(fieldeval), (delta[0], delta[1] + delta[3] * 0.22), connectionstyle="arc3,rad=0.18")

# ============================================================ mitigation ==
mit = box(51, 0.5, 75, 6.2,
          "Field fine-tuning mitigation (tomato, MobileNetV2): full vs. head-only vs. L2-SP\n"
          "15/cls train, 5/cls val, remainder test → re-eval on field + lab (§IV-F, V-E)",
          MIT_FILL, MIT_TXT, fontsize=7.9)
arrow((mit[0] + 12, mit[1] + mit[3]), (fp32[0] + 3, fp32[1]),
      connectionstyle="arc3,rad=0.18", ls="--", color=MIT_FILL, lw=1.3)

ax.set_ylim(0, 34.5)
fig.subplots_adjust(left=0.004, right=0.996, top=0.995, bottom=0.005)
fig.savefig("figures/architecture_pipeline_v3.png", dpi=260, bbox_inches="tight")
from PIL import Image
im = Image.open("figures/architecture_pipeline_v3.png")
print("saved", im.size, "aspect", im.size[0] / im.size[1])
