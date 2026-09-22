import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(11.5, 7.4))
ax.set_xlim(0, 116)
ax.set_ylim(0, 76)
ax.axis("off")

COL = {
    "data": "#dce6f2",
    "prep": "#e8e1f2",
    "model": "#1f4e8c",
    "quant": "#c0392b",
    "eval": "#2e7d32",
    "mit": "#b8860b",
    "text_dark": "#1a1a1a",
}

def box(x, y, w, h, text, fc, ec="#333333", fontsize=8.3, fontcolor="white", weight="normal", ls="-"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.6",
                        linewidth=1.1, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=2)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize,
             color=fontcolor, weight=weight, zorder=3, linespacing=1.35)
    return (x, y, w, h)

def arrow(p1, p2, style="-|>", color="#333333", lw=1.3, connectionstyle="arc3,rad=0.0", ls="-"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12, linewidth=lw,
                         color=color, connectionstyle=connectionstyle, zorder=1, linestyle=ls)
    ax.add_patch(a)

def right(bx): return (bx[0]+bx[2], bx[1]+bx[3]/2)
def left(bx): return (bx[0], bx[1]+bx[3]/2)
def top(bx): return (bx[0]+bx[2]/2, bx[1]+bx[3])
def bottom(bx): return (bx[0]+bx[2]/2, bx[1])
def topleft(bx): return (bx[0]+bx[2]*0.15, bx[1]+bx[3])
def topright(bx): return (bx[0]+bx[2]*0.85, bx[1]+bx[3])

# ---------- Column 1: Data sources ----------
pv = box(1, 58, 20, 10, "PlantVillage\n(lab, 21 classes,\n7 crops)\nHughes & Salathé 2015", COL["data"], fontcolor="#1a1a1a")
pd_ = box(1, 44, 20, 10, "PlantDoc\n(field, real-world\nphotographs)\nSingh et al. 2020", COL["data"], fontcolor="#1a1a1a")

# ---------- Column 2: preprocessing / split ----------
prep = box(26, 51, 21, 12, "Preprocessing\n96×96 resize, [-1,1]\nnormalize, flip/\nbrightness/contrast\naug. (train only)", COL["prep"], fontcolor="#1a1a1a")
arrow(right(pv), left(prep), connectionstyle="arc3,rad=-0.15")
arrow(right(pd_), left(prep), connectionstyle="arc3,rad=0.15")

split = box(51, 51, 20, 12, "70/15/15 split\n(pv_train / pv_val /\npv_test); all\nPlantDoc → field\nheld-out set", COL["prep"], fontcolor="#1a1a1a")
arrow(right(prep), left(split))

# ---------- Column 3: training ----------
train = box(75, 51, 24, 16, "Train from scratch\n(no ImageNet weights)\nMobileNetV2 • MobileNetV3-Small\nEfficientNetB0\nAdam, BN-momentum fix\n(Sec. IV-B, IV-E)", COL["model"], fontcolor="white")
arrow(right(split), left(train), connectionstyle="arc3,rad=0.08")

fp32 = box(88, 34, 24, 10, "FP32 trained model\n(×3 architectures\n×2 experiments)", COL["model"], fontcolor="white")
arrow(bottom(train), (fp32[0]+fp32[2]/2+4, fp32[1]+fp32[3]), connectionstyle="arc3,rad=0.0")

# ---------- quantization ----------
int8 = box(88, 19, 24, 10, "INT8 post-training\nquantization (TFLite,\n200-image calibration)\n(Sec. IV-C)", COL["quant"], fontcolor="white")
arrow(bottom(fp32), top(int8))

# ---------- evaluation branch ----------
labeval = box(60, 3, 22, 11, "Lab evaluation\n(pv_test, in-domain)\n→ Table II, VI", COL["eval"], fontcolor="white")
fieldeval = box(88, 3, 24, 11, "Field evaluation\n(PlantDoc, out-of-\ndomain) → Table II, VI\nconfusion matrix (Fig. 4)", COL["eval"], fontcolor="white")
arrow(left(int8), (labeval[0]+labeval[2], labeval[1]+labeval[3]), connectionstyle="arc3,rad=0.25")
arrow(bottom(int8), top(fieldeval))
arrow(right(fp32), (fieldeval[0]+fieldeval[2]-3, fieldeval[1]+fieldeval[3]+1), connectionstyle="arc3,rad=-0.35", ls="--", color="#666666")

domaingap = box(34, 3, 24, 11, "Δ_domain = Acc_lab − Acc_field\n(Sec. IV-D, Eq. 4)\nreported for every model /\nprecision / architecture", "#555555", fontcolor="white")
arrow(left(labeval), right(domaingap))

# ---------- mitigation branch (bottom-left) ----------
mit = box(1, 3, 31, 17, "Field fine-tuning mitigation\n(tomato, MobileNetV2 only)\nD_ft-train 15/cls, D_ft-val 5/cls,\nD_ft-test 584 img (disjoint)\nFull fine-tune (θ′=θ) vs.\nHead-only (θ′={W,b})\n→ re-eval on D_ft-test & pv_test\n(Sec. IV-F, V-E, Table VIII)", COL["mit"], fontcolor="white", fontsize=7.4)
arrow((mit[0]+mit[2], mit[1]+mit[3]*0.75), (fp32[0]+2, fp32[1]), connectionstyle="arc3,rad=-0.35", ls="--", color="#8a6d00")
ax.text(1, 22, "starts from trained\nFP32 tomato model →", fontsize=6.8, color="#8a6d00", style="italic")

# legend
legend_items = [
    ("Data", COL["data"], "#1a1a1a"), ("Preprocessing / split", COL["prep"], "#1a1a1a"),
    ("Model training", COL["model"], "white"), ("Quantization", COL["quant"], "white"),
    ("Evaluation", COL["eval"], "white"), ("Mitigation", COL["mit"], "white"),
]
lx = 1
for label, fc, tc in legend_items:
    ax.add_patch(FancyBboxPatch((lx, 70.5), 1.8, 1.8, boxstyle="round,pad=0.1,rounding_size=0.4", facecolor=fc, edgecolor="#333333", linewidth=0.8))
    ax.text(lx + 2.3, 71.4, label, fontsize=7.3, va="center")
    lx += len(label) * 0.62 + 5.5

fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.01)
fig.savefig("figures/architecture_pipeline.png", dpi=220, bbox_inches="tight")
from PIL import Image
print("saved", Image.open("figures/architecture_pipeline.png").size)
