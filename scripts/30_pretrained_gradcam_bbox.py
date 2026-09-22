"""
Same whole-test-set quantitative Grad-CAM bounding-box energy metric as
18_gradcam_bbox_energy.py / 26_cutmix_gradcam_bbox.py (see 18_ for the full
method), applied to the ImageNet-pretrained-backbone tomato model
(29_pretrained_finetune.py) instead of the from-scratch baseline. Tests
directly whether starting from real ImageNet features shifts attention
toward the leaf and away from background -- the mechanistic half of the
"tabula rasa confound" critique: even if pretraining narrows the accuracy
gap, does it fix (or fail to fix) the same background-fixation Section V-C
documents for the from-scratch model, exactly as CutMix (Section V-F) raised
accuracy without fixing attention?

Two things differ from 18_/26_'s preprocessing, both forced by the ported
backbone's calibration (see 28_convert_pretrained_weights.py): pixels are
normalized with (x/255 - mean)/std (torchvision-style), not x/127.5-1; and
the Grad-CAM target layer is 'block_13_expand_relu6' (this script's own
from-scratch-matched architecture naming), which sits at the same 6x6
feature-map resolution and the same relative depth (last feature map wider
than 3x3 at 96x96 input) as the baseline's 'block_13_expand_relu' --
verified below before scoring, not assumed.
"""
import csv, json, os, re
from collections import defaultdict
import numpy as np
import tensorflow as tf

IMG_SIZE = 96
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
model = tf.keras.models.load_model("model_tomato_pretrained_fp32.keras")
target_layer = model.get_layer(LAYER_NAME)
print(f"Grad-CAM target layer: {LAYER_NAME}, output shape {target_layer.output.shape} "
      f"(expect 6x6, matching the baseline/CutMix analysis's relative depth)")
assert target_layer.output.shape[1] == 6 and target_layer.output.shape[2] == 6, \
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

def preprocess(img96):
    return (img96 / 255.0 - MEAN) / STD

def gradcam_heatmap(img96, class_idx):
    x = np.expand_dims(preprocess(img96), 0).astype(np.float32)
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
    img96 = tf.image.resize(orig, [IMG_SIZE, IMG_SIZE]).numpy().astype(np.float32)

    true_orig = int(r["class_idx"])
    true_remapped = remap[true_orig]
    x = np.expand_dims(preprocess(img96), 0)
    preds = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    correct = pred_idx == true_remapped

    heatmap = gradcam_heatmap(img96, pred_idx)
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
json.dump(out, open("gradcam_bbox_energy_pretrained_results.json", "w"), indent=2)
print("Saved gradcam_bbox_energy_pretrained_results.json")
