"""
Quantitative Grad-CAM attention metric, addressing both external reviews'
request to replace the two-image qualitative Grad-CAM figure with a real,
whole-test-set number.

PlantDoc's *classification* release (Singh et al. 2020, used everywhere
else in this paper) ships no bounding boxes. But the same authors also
published a sibling *object-detection* release,
github.com/pratikkayal/PlantDoc-Object-Detection-Dataset, with per-image
leaf bounding boxes in train_labels.csv / test_labels.csv (Pascal-VOC-style
filename,width,height,class,xmin,ymin,xmax,ymax rows). Every one of this
paper's 744 tomato field-test images has a real, non-fabricated match in
that CSV by filename (verified: 744/744 direct match), so this script
computes a genuine per-image, whole-set metric rather than a synthetic one.

For every tomato field image:
  1. run the already-trained FP32 tomato model (same model, same
     block_13_expand_relu Grad-CAM layer as the existing qualitative
     figure, 16_gradcam.py) to get its prediction;
  2. compute the Grad-CAM heatmap at the predicted class and upsample it
     directly to the ORIGINAL image resolution (not the 96x96 model input)
     so it lines up with the bounding-box coordinates as annotated;
  3. union every annotated leaf box in that image into one region of
     interest (many field photos have more than one leaf boxed);
  4. compute energy_in_box = (heatmap mass inside the union of boxes) /
     (total heatmap mass), and chance = (union box area) / (image area) --
     the fraction we would expect by chance if attention were spread
     uniformly with no relation to the leaf at all.

This turns the qualitative "attention drifts off the leaf" claim into a
paired, whole-set number: mean/median energy_in_box vs. mean chance,
split by whether the field prediction was correct.
"""
import csv, json, os, re
from collections import defaultdict
import numpy as np
import tensorflow as tf

IMG_SIZE = 96
LAYER_NAME = "block_13_expand_relu"
BBOX_REPO = "/tmp/plantdoc_od_probe"
CLS_REPO_FIELD_ROOT = "/tmp/plantdoc_probe"

STOCK_PATTERNS = [
    re.compile(r"-[a-z0-9]{6,7}\.jpe?g$", re.I),
    re.compile(r"260nw|shutterstock|picture-id|gettyimages|istock", re.I),
    re.compile(r"leaf-\d{6,9}\.jpe?g$", re.I),
]
def looks_like_stock(path):
    fname = path.split("/")[-1]
    return any(p.search(fname) for p in STOCK_PATTERNS)

# ---- load bbox annotations, union boxes per filename --------------------
boxes_by_file = defaultdict(list)
dims_by_file = {}
for fn in ["train_labels.csv", "test_labels.csv"]:
    with open(os.path.join(BBOX_REPO, fn)) as f:
        for row in csv.DictReader(f):
            fname = row["filename"]
            boxes_by_file[fname].append(
                (int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"]))
            )
            dims_by_file[fname] = (int(row["width"]), int(row["height"]))

# ---- model + manifest -----------------------------------------------------
model = tf.keras.models.load_model("model_tomato_fp32.keras")
grad_model = tf.keras.models.Model(
    inputs=model.inputs, outputs=[model.get_layer(LAYER_NAME).output, model.output]
)
classmap = json.load(open("classmap_tomato.json"))
remap = {int(k): v for k, v in classmap["remap"].items()}

rows = list(csv.DictReader(open("manifest.csv")))
field_rows = [r for r in rows if r["is_tomato"] == "True" and r["domain"] == "pd_field"
              and not looks_like_stock(r["path"])]

def load_original(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img.numpy()  # HxWx3, original resolution

def gradcam_heatmap(img96, class_idx):
    x = np.expand_dims(img96 / 127.5 - 1.0, 0).astype(np.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()  # small, e.g. 6x6

def resize_heatmap(heatmap, out_h, out_w):
    hm = tf.expand_dims(tf.expand_dims(heatmap, 0), -1)
    hm = tf.image.resize(hm, [out_h, out_w], method="bilinear")
    return hm.numpy()[0, :, :, 0]

results = []
skipped = 0
for r in field_rows:
    fname = os.path.basename(r["path"])
    if fname not in boxes_by_file:
        skipped += 1
        continue
    orig = load_original(r["path"])
    H, W = orig.shape[0], orig.shape[1]
    img96 = tf.image.resize(orig, [IMG_SIZE, IMG_SIZE]).numpy().astype(np.float32)

    true_orig = int(r["class_idx"])
    true_remapped = remap[true_orig]
    x = np.expand_dims(img96 / 127.5 - 1.0, 0)
    preds = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    correct = pred_idx == true_remapped

    heatmap = gradcam_heatmap(img96, pred_idx)
    heatmap_full = resize_heatmap(heatmap, H, W)  # aligned to original image pixels

    mask = np.zeros((H, W), dtype=bool)
    box_area = 0
    for (xmin, ymin, xmax, ymax) in boxes_by_file[fname]:
        xmin, xmax = max(0, min(xmin, W)), max(0, min(xmax, W))
        ymin, ymax = max(0, min(ymin, H)), max(0, min(ymax, H))
        if xmax <= xmin or ymax <= ymin:
            continue
        mask[ymin:ymax, xmin:xmax] = True
    box_area = int(mask.sum())
    if box_area == 0:
        skipped += 1
        continue

    total_energy = float(heatmap_full.sum()) + 1e-9
    in_box_energy = float(heatmap_full[mask].sum())
    energy_frac = in_box_energy / total_energy
    chance_frac = box_area / (H * W)

    results.append({
        "file": fname, "correct": bool(correct),
        "energy_in_box": energy_frac, "chance": chance_frac,
        "n_boxes": len(boxes_by_file[fname]),
    })

print(f"Scored {len(results)} field images ({skipped} skipped: no usable bbox match)")

def summarize(subset, label):
    if not subset:
        return None
    e = np.array([s["energy_in_box"] for s in subset])
    c = np.array([s["chance"] for s in subset])
    diff = e - c
    rng = np.random.default_rng(7)
    boots = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"{label}: n={len(subset)}  mean energy_in_box={e.mean():.3f}  "
          f"mean chance={c.mean():.3f}  mean excess={diff.mean():+.3f} "
          f"[95% CI {lo:+.3f}, {hi:+.3f}]  median energy_in_box={np.median(e):.3f}")
    return {
        "n": len(subset), "mean_energy_in_box": float(e.mean()),
        "median_energy_in_box": float(np.median(e)), "mean_chance": float(c.mean()),
        "mean_excess": float(diff.mean()), "excess_ci95": [float(lo), float(hi)],
    }

all_summary = summarize(results, "All field images")
correct_subset = [s for s in results if s["correct"]]
wrong_subset = [s for s in results if not s["correct"]]
correct_summary = summarize(correct_subset, "Correctly classified field images")
wrong_summary = summarize(wrong_subset, "Misclassified field images")

out = {
    "n_scored": len(results), "n_skipped": skipped,
    "all": all_summary, "correct": correct_summary, "misclassified": wrong_summary,
    "per_image": results,
}
json.dump(out, open("gradcam_bbox_energy_results.json", "w"), indent=2)
print("Saved gradcam_bbox_energy_results.json")
