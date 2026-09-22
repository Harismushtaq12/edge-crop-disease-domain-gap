"""
Generate real figures from actual training logs, actual trained/quantized
models, and actual dataset images for the paper. Nothing here is a mockup --
every number/pixel comes from files produced by 03_train.py / 04_.../ 05_...
"""
import csv, json, re, os, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

random.seed(42)
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ---------- 1. Training curves (parsed from real training logs) ----------
def parse_log(path):
    epochs, tr_acc, val_acc, tr_loss, val_loss = [], [], [], [], []
    pat = re.compile(
        r"accuracy:\s*([\d.]+)\s*-\s*loss:\s*([\d.]+)\s*-\s*val_accuracy:\s*([\d.]+)\s*-\s*val_loss:\s*([\d.]+)"
    )
    for line in open(path):
        m = pat.search(line)
        if m:
            tr_acc.append(float(m.group(1)))
            tr_loss.append(float(m.group(2)))
            val_acc.append(float(m.group(3)))
            val_loss.append(float(m.group(4)))
            epochs.append(len(epochs) + 1)
    return epochs, tr_acc, val_acc, tr_loss, val_loss

fig, axes = plt.subplots(2, 2, figsize=(4.6, 4.2))
for i, (exp, title) in enumerate([("tomato", "Tomato (8-class)"), ("multicrop", "Multi-crop (21-class)")]):
    ep, tra, va, trl, vl = parse_log(f"train_{exp}.log")
    axes[i, 0].plot(ep, tra, label="train", color="#1f4e8c", linewidth=1)
    axes[i, 0].plot(ep, va, label="val", color="#c0392b", linewidth=1)
    axes[i, 0].set_ylabel(f"{title}\nAccuracy", fontsize=7)
    axes[i, 0].tick_params(labelsize=6); axes[i, 0].grid(alpha=0.3)
    axes[i, 1].plot(ep, trl, label="train", color="#1f4e8c", linewidth=1)
    axes[i, 1].plot(ep, vl, label="val", color="#c0392b", linewidth=1)
    axes[i, 1].set_ylabel("Loss", fontsize=7)
    axes[i, 1].tick_params(labelsize=6); axes[i, 1].grid(alpha=0.3)
    if i == 0:
        axes[i, 0].legend(fontsize=6); axes[i, 1].legend(fontsize=6)
    if i == 1:
        axes[i, 0].set_xlabel("Epoch", fontsize=7); axes[i, 1].set_xlabel("Epoch", fontsize=7)
fig.suptitle("Training history (real runs, from-scratch MobileNetV2)", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/train_curves_combined.png", dpi=220)
plt.close(fig)
print("train_curves_combined.png")

# ---------- 2. Per-class bar chart (tomato, lab vs field, INT8) ----------
pc = json.load(open("perclass_tomato.json"))
names = list(pc["field"].keys())
names.sort(key=lambda k: -pc["field"][k][0])
field_acc = [pc["field"][k][0] * 100 for k in names]
lab_acc = [pc["lab"][k][0] * 100 for k in names]
short = [n.replace("Tomato___", "").replace("_", " ") for n in names]

fig, ax = plt.subplots(figsize=(7.2, 3.2))
x = np.arange(len(names))
w = 0.38
ax.bar(x - w/2, lab_acc, w, label="Lab (PlantVillage test)", color="#1f4e8c")
ax.bar(x + w/2, field_acc, w, label="Field (PlantDoc)", color="#c0392b")
ax.set_xticks(x); ax.set_xticklabels(short, rotation=30, ha="right", fontsize=7)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Tomato per-class accuracy, INT8 model: lab vs. field", fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/perclass_tomato_bar.png", dpi=200)
plt.close(fig)
print("perclass_tomato_bar.png")

# ---------- 3. Confusion matrices (tomato INT8, lab + field) ----------
classmap = json.load(open("classmap_tomato.json"))
remap = {int(k): v for k, v in classmap["remap"].items()}
n_classes = len(classmap["class_ids"])
rows = [r for r in csv.DictReader(open("manifest.csv")) if r["is_tomato"] == "True"]
idx_to_name = {}
for r in rows:
    idx_to_name[int(r["class_idx"])] = r["class_name"].replace("Tomato___", "").replace("_", " ")
class_order = [idx_to_name[cid] for cid in sorted(set(int(r["class_idx"]) for r in rows))]

interpreter = tf.lite.Interpreter(model_path="model_tomato_int8.tflite")
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
IMG_SIZE = 96

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 127.5 - 1.0
    return img.numpy().astype(np.float32)

def confusion(rows_eval):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for r in rows_eval:
        img = load_img(r["path"])
        x = np.expand_dims(img, 0)
        scale, zero_point = inp["quantization"]
        x = (x / scale + zero_point).astype(np.int8)
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        pred = int(np.argmax(interpreter.get_tensor(out["index"])[0]))
        true = remap[int(r["class_idx"])]
        cm[true, pred] += 1
    return cm

field_rows = [r for r in rows if r["domain"] == "pd_field"]
lab_rows = [r for r in rows if r["domain"] == "pv_test"]
cm_field = confusion(field_rows)
cm_lab = confusion(lab_rows)
np.savez(f"{OUT}/confusion_tomato.npz", field=cm_field, lab=cm_lab, labels=class_order)

def draw_cm(ax, cm, title):
    cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_order, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(class_order, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_title(title, fontsize=9)
    for i in range(n_classes):
        for j in range(n_classes):
            if cmn[i, j] > 0.01:
                ax.text(j, i, f"{cmn[i,j]*100:.0f}", ha="center", va="center",
                        fontsize=6, color="white" if cmn[i, j] > 0.5 else "black")
    return im

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
draw_cm(axes[0], cm_lab, "(a) Lab (PlantVillage test)")
axes[0].set_ylabel("True", fontsize=8)
im = draw_cm(axes[1], cm_field, "(b) Field (PlantDoc)")
fig.suptitle("Tomato INT8 model: confusion matrices (row-normalized %)", fontsize=10)
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
fig.savefig(f"{OUT}/confmat_tomato_combined.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("confmat_tomato_combined.png")

# ---------- 4. Real lab-vs-field sample image grid ----------
SAMPLE_CLASSES = [
    ("Tomato___Early_blight", "Tomato Early blight leaf", "Early blight"),
    ("Tomato___healthy", "Tomato leaf", "Healthy"),
    ("Tomato___Bacterial_spot", "Tomato leaf bacterial spot", "Bacterial spot"),
    ("Tomato___Tomato_mosaic_virus", "Tomato leaf mosaic virus", "Mosaic virus"),
]
PV_ROOT = "/tmp/plantvillage_test/raw/color"
PD_ROOT = "/tmp/plantdoc_test"

fig, axes = plt.subplots(2, len(SAMPLE_CLASSES), figsize=(7.2, 3.6))
for j, (pv, pd, label) in enumerate(SAMPLE_CLASSES):
    pv_dir = os.path.join(PV_ROOT, pv)
    pv_file = sorted(os.listdir(pv_dir))[3]
    img = mpimg.imread(os.path.join(pv_dir, pv_file))
    axes[0, j].imshow(img); axes[0, j].axis("off")
    axes[0, j].set_title(label, fontsize=8)

    pd_dir = os.path.join(PD_ROOT, "test", pd)
    if not os.path.isdir(pd_dir) or not os.listdir(pd_dir):
        pd_dir = os.path.join(PD_ROOT, "train", pd)
    pd_files = sorted(os.listdir(pd_dir))
    pick = {"Tomato___Tomato_mosaic_virus": "tmv-tomato-9.jpg"}.get(pv)
    pd_file = pick if pick in pd_files else pd_files[1]
    img2 = mpimg.imread(os.path.join(pd_dir, pd_file))
    axes[1, j].imshow(img2); axes[1, j].axis("off")

axes[0, 0].set_ylabel("Lab\n(PlantVillage)", fontsize=8)
axes[1, 0].set_ylabel("Field\n(PlantDoc)", fontsize=8)
for ax, lab in zip([axes[0, 0], axes[1, 0]], ["Lab (PlantVillage)", "Field (PlantDoc)"]):
    ax.axis("on")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(lab, fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.suptitle("Real training (lab) vs. evaluation (field) images for the same classes", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/lab_vs_field_samples.png", dpi=200)
plt.close(fig)
print("lab_vs_field_samples.png")

print("ALL FIGURES DONE ->", os.listdir(OUT))
