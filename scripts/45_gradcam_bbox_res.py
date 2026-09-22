"""
Grad-CAM leaf-box energy for any checkpoint in the ported-ImageNet family.

Parameterised form of 30_pretrained_gradcam_bbox.py so the stacked
(pretrained + L2-SP) model is scored by the *identical* protocol and
target layer as the pretrained-only model, which is what makes the
three-way mechanism comparison in the paper legitimate.

Usage: python3 45_gradcam_bbox_res.py <model.keras> <out.json> [img_size]
"""
import csv, json, os, re, sys
from collections import defaultdict
import numpy as np
import tensorflow as tf

IMG_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 96
LAYER_NAME = "block_13_expand_relu6"
BBOX_REPO = "/tmp/plantdoc_od_probe"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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
for fn in ["train_labels.csv", "test_labels.csv"]:
    with open(os.path.join(BBOX_REPO, fn)) as f:
        for row in csv.DictReader(f):
            fname = row["filename"]
            boxes_by_file[fname].append(
                (int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"]))
            )

# ---- model + manifest -----------------------------------------------------
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "model_tomato_pretrained_fp32.keras"
OUT_JSON = sys.argv[2] if len(sys.argv) > 2 else "gradcam_bbox_energy_pretrained_results.json"
print("scoring:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
target_layer = model.get_layer(LAYER_NAME)
# The SAME named layer is read at every resolution. Its feature map is 6x6 at
# 96px input and 14x14 at 224px -- same network depth and semantic level, more
# spatial detail. Reading the same depth is what makes the comparison a
# resolution comparison rather than a depth comparison; the metric itself is a
# normalised energy fraction, so it stays comparable across grid sizes.
expected = max(1, round(IMG_SIZE / 16))
print(f"Grad-CAM target layer: {LAYER_NAME}, output shape {target_layer.output.shape} "
      f"at {IMG_SIZE}px input (expected ~{expected}x{expected})")
assert target_layer.output.shape[1] == target_layer.output.shape[2] == expected, \
    "target layer resolution does not match the baseline analysis -- pick a different layer"

grad_model = tf.keras.models.Model(
    inputs=model.inputs, outputs=[target_layer.output, model.output]
)
classmap = json.load(open("classmap_tomato.json"))
remap = {int(k): v for k, v in classmap["remap"].items()}

rows = list(csv.DictReader(open("manifest.csv")))
field_rows = [r for r in rows if r["is_tomato"] == "True" and r["domain"] == "pd_field"
              and not looks_like_stock(r["path"])]

def load_original(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img.numpy()

def preprocess(img_in):
    return (img_in / 255.0 - MEAN) / STD

def gradcam_heatmap(img_in, class_idx):
    x = np.expand_dims(preprocess(img_in), 0).astype(np.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

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
    img_in = tf.image.resize(orig, [IMG_SIZE, IMG_SIZE]).numpy().astype(np.float32)

    true_orig = int(r["class_idx"])
    true_remapped = remap[true_orig]
    x = np.expand_dims(preprocess(img_in), 0)
    preds = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    correct = pred_idx == true_remapped

    heatmap = gradcam_heatmap(img_in, pred_idx)
    heatmap_full = resize_heatmap(heatmap, H, W)

    mask = np.zeros((H, W), dtype=bool)
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
json.dump(out, open(OUT_JSON, "w"), indent=2)
print("Saved", OUT_JSON)
