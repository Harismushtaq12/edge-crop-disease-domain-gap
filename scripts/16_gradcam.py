"""
Grad-CAM visualization: turns the confusion-matrix "attractor class" finding
(model over-predicts early blight / healthy on field images regardless of
true class) into visible evidence of *where* the model is looking.

Story the figure tells, left to right:
  1. Lab image, true=Early blight, correctly classified -> Grad-CAM lights
     up the actual lesions. The model DID learn a real lesion feature.
  2. Field image, true=something else, WRONGLY predicted "Early blight" ->
     Grad-CAM at the predicted class shows what fired instead -- if it is
     not lesion-shaped, that is the failure made visible.
  3. Lab image, true=healthy, correctly classified -> Grad-CAM shows a
     diffuse, low-magnitude, no-lesion pattern (correctly "nothing here").
  4. Field image, true=something else, WRONGLY predicted "healthy" ->
     Grad-CAM at the predicted class shows what the model treated as
     evidence of absence-of-disease on a real diseased leaf.

Uses the FP32 Keras tomato model (Grad-CAM needs gradients; the INT8 TFLite
export does not expose them, and the paper's own Table II shows FP32 and
INT8 accuracy track closely, so FP32 attention is a fair proxy). Computed
at block_13_expand_relu (6x6x432) -- the last layer with more than 3x3
spatial resolution, for finer localization than the network's final layer.
Both remain coarse because of the 96x96 edge-deployment input size; that is
an inherent trade-off of small inputs for on-device inference, not a
Grad-CAM artifact, and is disclosed as such in the paper text.
"""
import csv, json, re
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

IMG_SIZE = 96
LAYER_NAME = "block_13_expand_relu"

STOCK_PATTERNS = [
    re.compile(r"-[a-z0-9]{6,7}\.jpe?g$", re.I),
    re.compile(r"260nw|shutterstock|picture-id|gettyimages|istock", re.I),
    re.compile(r"leaf-\d{6,9}\.jpe?g$", re.I),
]
def looks_like_stock(path):
    fname = path.split("/")[-1]
    return any(p.search(fname) for p in STOCK_PATTERNS)

model = tf.keras.models.load_model("model_tomato_fp32.keras")
grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[model.get_layer(LAYER_NAME).output, model.output],
)

classmap = json.load(open("classmap_tomato.json"))
remap = {int(k): v for k, v in classmap["remap"].items()}
inv_remap = {v: k for k, v in remap.items()}
rows = list(csv.DictReader(open("manifest.csv")))
rows = [r for r in rows if r["is_tomato"] == "True"]
idx_to_name = {}
for r in rows:
    idx_to_name[int(r["class_idx"])] = r["class_name"].split("___")[-1].replace("_", " ")
name_by_remapped = {remap[k]: v for k, v in idx_to_name.items()}

EARLY_BLIGHT_CLASS = 14   # original class_idx
HEALTHY_CLASS = 18
EARLY_BLIGHT_REMAPPED = remap[EARLY_BLIGHT_CLASS]
HEALTHY_REMAPPED = remap[HEALTHY_CLASS]

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img.numpy().astype(np.float32)

def predict_fp32(img_arr):
    x = np.expand_dims(img_arr / 127.5 - 1.0, 0)
    preds = model.predict(x, verbose=0)[0]
    return int(np.argmax(preds)), preds

def gradcam(img_arr, class_idx):
    x = np.expand_dims(img_arr / 127.5 - 1.0, 0).astype(np.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay(img_arr, heatmap, alpha=0.42):
    hm = np.uint8(255 * heatmap)
    hm_img = tf.keras.utils.array_to_img(np.expand_dims(hm, -1))
    hm_img = hm_img.resize((IMG_SIZE, IMG_SIZE))
    hm_resized = np.array(hm_img)
    jet = cm.get_cmap("jet")(hm_resized / 255.0)[:, :, :3] * 255
    base = img_arr.astype(np.uint8)
    blended = (alpha * jet + (1 - alpha) * base).astype(np.uint8)
    return blended

lab_rows = [r for r in rows if r["domain"] == "pv_test" and not looks_like_stock(r["path"])]
field_rows = [r for r in rows if r["domain"] == "pd_field" and not looks_like_stock(r["path"])]

rng = np.random.default_rng(3)

def find_case(row_pool, want_true_class=None, want_pred_class=None, exclude_true_class=None):
    pool = list(row_pool)
    rng.shuffle(pool)
    for r in pool:
        true_orig = int(r["class_idx"])
        true_remapped = remap[true_orig]
        if want_true_class is not None and true_orig != want_true_class:
            continue
        if exclude_true_class is not None and true_orig == exclude_true_class:
            continue
        img_arr = load_img(r["path"])
        pred_idx, preds = predict_fp32(img_arr)
        if want_pred_class is not None and pred_idx != want_pred_class:
            continue
        if want_true_class is not None and pred_idx != true_remapped:
            continue  # for the "correct lab example" cases, require it be correct
        return r, img_arr, pred_idx, true_remapped
    return None

# Case 1: lab, true=Early blight, correctly classified.
c1 = find_case(lab_rows, want_true_class=EARLY_BLIGHT_CLASS)
# Case 2: field, true != Early blight, wrongly predicted Early blight.
c2 = find_case(field_rows, exclude_true_class=EARLY_BLIGHT_CLASS, want_pred_class=EARLY_BLIGHT_REMAPPED)
# Case 3: lab, true=healthy, correctly classified.
c3 = find_case(lab_rows, want_true_class=HEALTHY_CLASS)
# Case 4: field, true != healthy, wrongly predicted healthy.
c4 = find_case(field_rows, exclude_true_class=HEALTHY_CLASS, want_pred_class=HEALTHY_REMAPPED)

cases = [
    (c1, "Lab, true: Early blight\n(correctly classified)"),
    (c2, "Field, true: {true}\n(wrongly → Early blight)"),
    (c3, "Lab, true: Healthy\n(correctly classified)"),
    (c4, "Field, true: {true}\n(wrongly → Healthy)"),
]

fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
for col, (case, title_tpl) in enumerate(cases):
    r, img_arr, pred_idx, true_remapped = case
    true_name = name_by_remapped[true_remapped]
    pred_name = name_by_remapped[pred_idx]
    title = title_tpl.format(true=true_name)
    heatmap = gradcam(img_arr, pred_idx)
    blended = overlay(img_arr, heatmap)

    axes[0, col].imshow(img_arr.astype(np.uint8))
    axes[0, col].set_title(title, fontsize=8.6)
    axes[0, col].axis("off")

    axes[1, col].imshow(blended)
    correct = pred_idx == true_remapped
    color = "#1b7a1b" if correct else "#b3261e"
    mark = ("✓ pred: " + pred_name) if correct else ("✗ pred: " + pred_name)
    axes[1, col].set_title(f"Grad-CAM at predicted class\n{mark}", fontsize=8.3, color=color)
    axes[1, col].axis("off")

fig.suptitle("Grad-CAM (block_13, FP32 model): real lesion attention in the lab vs. the attractor-class misfire in the field", fontsize=10, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("figures/gradcam_field.png", dpi=200)
plt.close(fig)
print("Saved figures/gradcam_field.png")
for (case, _), name in zip(cases, ["lab-EB-correct", "field-wrong-to-EB", "lab-healthy-correct", "field-wrong-to-healthy"]):
    r, img_arr, pred_idx, true_remapped = case
    print(name, "path:", r["path"], "true:", name_by_remapped[true_remapped], "pred:", name_by_remapped[pred_idx])
