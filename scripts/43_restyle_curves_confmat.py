"""
Restyles the last two figures still carrying matplotlib defaults: the
training curves and the lab/field confusion matrices.

Both read exactly the data their predecessors read -- the raw training logs
and the saved confusion counts -- so only the encoding changes. Two specific
fixes beyond the shared style: the confusion matrices now use a single-hue
sequential ramp (a rainbow map makes ordered magnitudes unreadable and
survives neither greyscale printing nor colour-vision deficiency), and cells
carry their values directly so the colour is a redundant channel rather than
the only one.

Produces:
  figures/train_curves_v2.png
  figures/confmat_v2.png
"""
import re
import numpy as np
import figstyle as fs
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

fs.apply()

# ------------------------------------------------------------ training curves
LOG_RE = re.compile(
    r"accuracy:\s*([0-9.]+)\s*-\s*loss:\s*([0-9.]+)\s*-\s*"
    r"val_accuracy:\s*([0-9.]+)\s*-\s*val_loss:\s*([0-9.]+)")


def parse(path):
    acc, loss, vacc, vloss = [], [], [], []
    for line in open(path):
        m = LOG_RE.search(line)
        if m:
            acc.append(float(m.group(1)))
            loss.append(float(m.group(2)))
            vacc.append(float(m.group(3)))
            vloss.append(float(m.group(4)))
    return np.array(acc), np.array(loss), np.array(vacc), np.array(vloss)


RUNS = [("train_tomato.log", "Tomato, 8 classes"),
        ("train_multicrop.log", "Multi-crop, 21 classes")]

fig, axes = plt.subplots(2, 2, figsize=(fs.FULL_W, 3.9), sharex="col")
for r, (path, title) in enumerate(RUNS):
    acc, loss, vacc, vloss = parse(path)
    ep = np.arange(1, len(acc) + 1)

    ax = axes[r][0]
    ax.plot(ep, acc * 100, color=fs.BLUE, label="train")
    ax.plot(ep, vacc * 100, color=fs.ORANGE, label="validation")
    ax.set_ylabel("Accuracy (%)", fontsize=fs.FS_LABEL - 0.5)
    ax.set_title(title, loc="right", fontsize=fs.FS_TITLE - 1.0,
                 fontweight="bold", color=fs.INK_2, pad=4)
    ax.set_ylim(0, 102)
    ax.grid(True)
    if r == 0:
        ax.legend(loc="lower right", fontsize=fs.FS_LEGEND)
        fs.panel_title(ax, "a", "Accuracy")

    ax2 = axes[r][1]
    ax2.plot(ep, loss, color=fs.BLUE, label="train")
    ax2.plot(ep, vloss, color=fs.ORANGE, label="validation")
    ax2.set_ylabel("Loss", fontsize=fs.FS_LABEL - 0.5)
    ax2.grid(True)
    if r == 0:
        fs.panel_title(ax2, "b", "Loss")

    if r == len(RUNS) - 1:
        ax.set_xlabel("Epoch")
        ax2.set_xlabel("Epoch")

fs.finish(fig, "figures/train_curves_v2.png")

# ------------------------------------------------------- confusion matrices
d = np.load("figures/confusion_tomato.npz")
labels = [str(x) for x in d["labels"]]
# the saved labels are already human-readable but too long for an 8x8 grid
PRETTY = {
    "Bacterial spot": "Bact. spot",
    "Septoria leaf spot": "Septoria",
    "Tomato mosaic virus": "Mosaic",
    "Tomato Yellow Leaf Curl Virus": "Yellow curl",
    "healthy": "Healthy",
    "Leaf Mold": "Leaf mold",
}
names = [PRETTY.get(l, l) for l in labels]

cmap = LinearSegmentedColormap.from_list("seqblue", fs.SEQ_BLUE)

fig, axes = plt.subplots(1, 2, figsize=(fs.FULL_W, 3.15))
for ax, key, panel, title in [(axes[0], "lab", "a", "Lab (PlantVillage)"),
                              (axes[1], "field", "b", "Field (PlantDoc)")]:
    M = d[key].astype(float)
    M = M / np.clip(M.sum(axis=1, keepdims=True), 1, None)   # row-normalised
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=fs.FS_TICK - 1.4)
    ax.set_yticklabels(names if panel == "a" else [""] * len(names),
                       fontsize=fs.FS_TICK - 1.4)
    ax.set_xlabel("Predicted", fontsize=fs.FS_LABEL - 0.5)
    if panel == "a":
        ax.set_ylabel("True", fontsize=fs.FS_LABEL - 0.5)
    fs.panel_title(ax, panel, title)
    ax.grid(False)
    # values in-cell: colour becomes a redundant channel, not the only one
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                        fontsize=fs.FS_ANNOT - 1.8,
                        color="#ffffff" if v > 0.55 else fs.INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

cb = fig.colorbar(im, ax=axes, fraction=0.028, pad=0.02)
cb.set_label("Row-normalised share of true class (%)", fontsize=fs.FS_ANNOT)
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb.set_ticklabels(["0", "25", "50", "75", "100"])
cb.ax.tick_params(labelsize=fs.FS_TICK - 1, length=0)
cb.outline.set_visible(False)

fs.finish(fig, "figures/confmat_v2.png", tight=False)
