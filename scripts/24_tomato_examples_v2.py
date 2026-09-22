"""
v2 of the lab-vs-field example figure (paper Fig. 2): expands from 4 to
all 8 tomato classes used in the tomato-only experiment, and is sized to
render at full column width in the paper instead of the illegible
fraction-of-textwidth it was placed at before. Real PlantVillage (lab) and
PlantDoc (field) images for every tomato class, not a subset.
"""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

STOCK_PATTERNS = [
    re.compile(r"-[a-z0-9]{6,7}\.jpe?g$", re.I),
    re.compile(r"260nw|shutterstock|picture-id|gettyimages|istock", re.I),
    re.compile(r"leaf-\d{6,9}\.jpe?g$", re.I),
]
def looks_like_stock(fname):
    return any(p.search(fname) for p in STOCK_PATTERNS)

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

fig, axes = plt.subplots(2, len(SAMPLE_CLASSES), figsize=(15.0, 4.35))
for j, (pv, pd, label, pd_pick) in enumerate(SAMPLE_CLASSES):
    pv_dir = os.path.join(PV_ROOT, pv)
    pv_files = sorted(f for f in os.listdir(pv_dir) if f.lower().endswith((".jpg", ".jpeg")))
    pv_file = pv_files[len(pv_files) // 3]
    img = mpimg.imread(os.path.join(pv_dir, pv_file))
    axes[0, j].imshow(img)
    axes[0, j].axis("off")
    axes[0, j].set_title(label, fontsize=9.4)

    pd_dir = os.path.join(PD_ROOT, "test", pd)
    if not os.path.isfile(os.path.join(pd_dir, pd_pick)):
        pd_dir = os.path.join(PD_ROOT, "train", pd)
    img2 = mpimg.imread(os.path.join(pd_dir, pd_pick))
    axes[1, j].imshow(img2)
    axes[1, j].axis("off")

for ax, lab in zip([axes[0, 0], axes[1, 0]], ["Lab\n(PlantVillage)", "Field\n(PlantDoc)"]):
    ax.axis("on")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(lab, fontsize=9.6)
    for spine in ax.spines.values():
        spine.set_visible(False)

fig.suptitle("All 8 tomato classes: real lab training images (top) vs. real field evaluation images (bottom)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("figures/lab_vs_field_samples_v2.png", dpi=220, bbox_inches="tight")
plt.close(fig)
from PIL import Image
im = Image.open("figures/lab_vs_field_samples_v2.png")
print("saved", im.size, "aspect", im.size[0] / im.size[1])
