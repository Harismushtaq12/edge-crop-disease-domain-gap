import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("figures/confusion_tomato.npz", allow_pickle=True)
cm_lab, cm_field, class_order = d["lab"], d["field"], list(d["labels"])
n_classes = len(class_order)

def draw_cm(ax, cm, title, show_ylabels):
    cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_order, rotation=45, ha="right", fontsize=8)
    if show_ylabels:
        ax.set_yticklabels(class_order, fontsize=8)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_title(title, fontsize=10)
    for i in range(n_classes):
        for j in range(n_classes):
            if cmn[i, j] > 0.01:
                ax.text(j, i, f"{cmn[i,j]*100:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if cmn[i, j] > 0.5 else "black")
    return im

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), gridspec_kw={"wspace": 0.12})
draw_cm(axes[0], cm_lab, "(a) Lab (PlantVillage test)", show_ylabels=True)
axes[0].set_ylabel("True", fontsize=9)
im = draw_cm(axes[1], cm_field, "(b) Field (PlantDoc)", show_ylabels=False)
axes[1].set_ylabel("True (same class order as (a))", fontsize=8)
fig.suptitle("Tomato INT8 model: confusion matrices (row-normalized %)", fontsize=11, y=0.99)
cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
cbar.ax.tick_params(labelsize=8)
fig.subplots_adjust(left=0.16, right=0.86, bottom=0.28, top=0.90)
fig.savefig("figures/confmat_tomato_combined.png", dpi=200)
plt.close(fig)
from PIL import Image
print("saved", Image.open("figures/confmat_tomato_combined.png").size)
