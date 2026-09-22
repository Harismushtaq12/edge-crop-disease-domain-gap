"""
Regenerates the paper's remaining data figures in the shared visual system
(figstyle.py), so all of them read as one designed set rather than a dozen
separately-improvised matplotlib defaults.

Nothing about the underlying numbers changes -- every figure here reads the
same result JSONs the previous versions read. What changes is the encoding:
one palette used in fixed slot order, direct value labels everywhere colour
alone would have carried meaning (so the figures survive greyscale printing
and colour-vision deficiency), recessive grid and axis chrome, consistent
type scale, and headroom reserved so no label ever collides with a mark.

Produces:
  figures/perclass_v2.png          per-class lab vs field accuracy
  figures/multiarch_v2.png         three backbones x two precisions
  figures/layer_sensitivity_v2.png block-level INT8 sensitivity sweep
"""
import json
import numpy as np
import figstyle as fs
import matplotlib.pyplot as plt

fs.apply()

PRETTY = {
    "Tomato___Bacterial_spot": "Bacterial spot",
    "Tomato___Early_blight": "Early blight",
    "Tomato___Late_blight": "Late blight",
    "Tomato___Leaf_Mold": "Leaf mold",
    "Tomato___Septoria_leaf_spot": "Septoria leaf spot",
    "Tomato___Tomato_mosaic_virus": "Mosaic virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Yellow leaf curl",
    "Tomato___healthy": "Healthy",
}

# ------------------------------------------------------- 1. per-class accuracy
pc = json.load(open("perclass_tomato.json"))
lab, field = pc["lab"], pc["field"]


def acc(entry):
    """Per-class records are stored as [accuracy, n_correct, n_total]."""
    if isinstance(entry, (list, tuple)):
        return float(entry[0]) * 100
    if isinstance(entry, dict):
        for k in ("accuracy", "acc"):
            if k in entry:
                return float(entry[k]) * 100
    return float(entry) * 100


classes = sorted(lab.keys(), key=lambda c: -acc(field[c]))
lv = [acc(lab[c]) for c in classes]
fv = [acc(field[c]) for c in classes]
names = [PRETTY.get(c, c.replace("Tomato___", "").replace("_", " ")) for c in classes]

fig, ax = plt.subplots(figsize=(fs.FULL_W, 2.45))
x = np.arange(len(classes))
w = 0.38
b1 = ax.bar(x - w / 2, lv, w, color=fs.C_LAB, label="Lab (PlantVillage)", zorder=3)
b2 = ax.bar(x + w / 2, fv, w, color=fs.C_FIELD, label="Field (PlantDoc)", zorder=3)
ax.axhline(12.5, color=fs.MUTED, lw=0.9, ls=(0, (4, 2.5)), zorder=2)
ax.text(len(classes) - 0.45, 14.0, "chance", fontsize=fs.FS_ANNOT - 0.3,
        color=fs.MUTED, ha="right", va="bottom")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=18, ha="right")
ax.set_ylabel("Top-1 accuracy (%)")
ax.set_ylim(0, 112)
fs.bar_labels(ax, b1, fmt="{:.0f}", fontsize=fs.FS_ANNOT - 0.8)
fs.bar_labels(ax, b2, fmt="{:.0f}", fontsize=fs.FS_ANNOT - 0.8)
fs.grid_y_only(ax)
fs.tick_labels(ax)
ax.legend(loc="upper right", ncol=2)
ax.set_title("Per-class collapse is uneven: the best lab class is among the worst in the field",
             loc="left", fontsize=fs.FS_TITLE, color=fs.INK, fontweight="bold")
fs.finish(fig, "figures/perclass_v2.png")

# ------------------------------------------------------ 2. multi-architecture
ARCHS = [("mobilenetv2", "MobileNetV2"),
         ("mobilenetv3small", "MobileNetV3-Small"),
         ("efficientnetb0", "EfficientNetB0")]
rows = []
for tag, label in ARCHS:
    fp = json.load(open(f"results_tomato_{tag}_fp32.json"))
    q = json.load(open(f"results_tomato_{tag}_quantized.json"))

    def g(d, key):
        """FP32 files nest accuracy under pv_test/pd_field; INT8 files are flat."""
        v = d.get(key)
        if isinstance(v, dict):
            return float(v["accuracy"]) * 100
        if v is not None:
            return float(v) * 100
        raise KeyError(key)

    rows.append({
        "label": label,
        "lab_fp32": g(fp, "pv_test"), "field_fp32": g(fp, "pd_field"),
        "lab_int8": g(q, "int8_lab_accuracy"), "field_int8": g(q, "int8_field_accuracy"),
    })

fig, ax = plt.subplots(figsize=(fs.FULL_W, 2.45))
x = np.arange(len(rows))
w = 0.2
series = [
    ("lab_fp32", "Lab, FP32", fs.BLUE),
    ("lab_int8", "Lab, INT8", "#9ec5f4"),
    ("field_fp32", "Field, FP32", fs.ORANGE),
    ("field_int8", "Field, INT8", "#f7b79a"),
]
for i, (key, lbl, col) in enumerate(series):
    off = (i - 1.5) * w
    b = ax.bar(x + off, [r[key] for r in rows], w * 0.92, color=col, label=lbl, zorder=3)
    fs.bar_labels(ax, b, fmt="{:.1f}", fontsize=fs.FS_ANNOT - 1.2)

ax.axhline(12.5, color=fs.MUTED, lw=0.9, ls=(0, (4, 2.5)), zorder=2)
ax.text(len(rows) - 0.55, 18.5, "chance", fontsize=fs.FS_ANNOT - 0.3,
        color=fs.MUTED, ha="right", va="bottom")
ax.set_xticks(x)
ax.set_xticklabels([r["label"] for r in rows])
ax.set_ylabel("Top-1 accuracy (%)")
ax.set_ylim(0, 105)
fs.grid_y_only(ax)
fs.tick_labels(ax)
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
ax.set_title("Field accuracy sits near chance for every backbone; only MobileNetV3-Small breaks under INT8",
             loc="left", fontsize=fs.FS_TITLE, color=fs.INK, fontweight="bold", pad=14)
fs.finish(fig, "figures/multiarch_v2.png")

# --------------------------------------------------- 3. layer sensitivity sweep
ls = json.load(open("layer_sensitivity_blocks_results.json"))
base = float(ls["baseline_fp32_acc"]) * 100
groups = ls["groups"]
gnames = list(groups.keys())
# each group record carries the accuracy with only that group quantized
drop = [float(groups[k]["drop"]) * 100 for k in gnames]
order = np.argsort(drop)[::-1]
gnames = [gnames[i] for i in order]
drop = [drop[i] for i in order]

fig, ax = plt.subplots(figsize=(fs.COL_W * 1.42, 2.25))
cols = [fs.RED if d > 5 else fs.C_BEFORE for d in drop]
b = ax.barh(np.arange(len(gnames))[::-1], drop, height=0.62, color=cols, zorder=3)
for y, d in zip(np.arange(len(gnames))[::-1], drop):
    ax.text(d + 0.8, y, f"{d:.1f}", va="center", ha="left",
            fontsize=fs.FS_ANNOT - 0.4, color=fs.INK)
ax.set_yticks(np.arange(len(gnames))[::-1])
ax.set_yticklabels(gnames, fontsize=fs.FS_TICK)
ax.set_xlabel("Lab accuracy lost when only this group is INT8 (pp)")
ax.set_xlim(0, max(drop) * 1.22)
ax.grid(True, axis="x")
ax.grid(False, axis="y")
fs.tick_labels(ax)
ax.set_title("The two-layer stem alone reproduces the whole INT8 collapse",
             loc="left", fontsize=fs.FS_TITLE, color=fs.INK, fontweight="bold")
fs.finish(fig, "figures/layer_sensitivity_v2.png")
