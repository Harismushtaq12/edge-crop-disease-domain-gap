"""
v3 of the lab-vs-field sample figure.

Two changes from v2, both for the journal version: the embedded title comes
off (the caption carries it, and a title inside the artwork duplicates the
caption in print), and every panel is a square centre crop so the two rows
align on a common grid instead of each column setting its own height.

Real PlantVillage (lab) and PlantDoc (field) images, all eight tomato
classes, no stock photography.
"""
import os
import numpy as np
import matplotlib.image as mpimg
import figstyle as fs

fs.apply()

SAMPLE_CLASSES = [
    ("Tomato___Early_blight", "Tomato Early blight leaf", "Early\nblight", "1234080-Early-Blight.jpg"),
    ("Tomato___healthy", "Tomato leaf", "Healthy", "Tomato59.JPG.jpg"),
    ("Tomato___Bacterial_spot", "Tomato leaf bacterial spot", "Bacterial\nspot", "Tom_Bact7.jpg"),
    ("Tomato___Late_blight", "Tomato leaf late blight", "Late\nblight", "110822-206-Tomato-blight.jpg"),
    ("Tomato___Leaf_Mold", "Tomato mold leaf", "Leaf\nmold", "black_leaf_mold_in_zina_tina.jpg"),
    ("Tomato___Septoria_leaf_spot", "Tomato Septoria leaf spot", "Septoria\nleaf spot", "6134794031202304.jpeg.jpg"),
    ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato leaf yellow virus", "Yellow leaf\ncurl virus", "8226402877_9abf151b5b_b.jpg"),
    ("Tomato___Tomato_mosaic_virus", "Tomato leaf mosaic virus", "Mosaic\nvirus", "9511.img.jpg"),
]
PV_ROOT = "/tmp/plantvillage_probe/raw/color"
PD_ROOT = "/tmp/plantdoc_probe"


def square(img):
    """Centre crop to a square, so every panel shares one aspect ratio."""
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


n = len(SAMPLE_CLASSES)
fig, axes = fs.figure(width=fs.FULL_W, height=fs.FULL_W / n * 2 + 0.42,
                      nrows=2, ncols=n)

for j, (pv, pd, label, pd_pick) in enumerate(SAMPLE_CLASSES):
    pv_dir = os.path.join(PV_ROOT, pv)
    pv_files = sorted(f for f in os.listdir(pv_dir)
                      if f.lower().endswith((".jpg", ".jpeg")))
    img = mpimg.imread(os.path.join(pv_dir, pv_files[len(pv_files) // 3]))
    axes[0, j].imshow(square(img))
    axes[0, j].set_title(label, fontsize=fs.FS_ANNOT, fontweight="normal",
                         color=fs.INK, pad=3, linespacing=1.25)

    pd_dir = os.path.join(PD_ROOT, "test", pd)
    if not os.path.isfile(os.path.join(pd_dir, pd_pick)):
        pd_dir = os.path.join(PD_ROOT, "train", pd)
    img2 = mpimg.imread(os.path.join(pd_dir, pd_pick))
    axes[1, j].imshow(square(img2))

for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

for ax, lab in zip([axes[0, 0], axes[1, 0]],
                   ["Lab\n(PlantVillage)", "Field\n(PlantDoc)"]):
    ax.set_ylabel(lab, fontsize=fs.FS_ANNOT, color=fs.INK_2, linespacing=1.3)

fig.subplots_adjust(left=0.035, right=0.999, top=0.90, bottom=0.005,
                    wspace=0.05, hspace=0.05)
fs.finish(fig, "figures/lab_vs_field_samples_v3.png", tight=False, dpi=300)
