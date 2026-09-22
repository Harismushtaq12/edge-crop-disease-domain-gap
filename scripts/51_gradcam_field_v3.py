"""
v3 of the Grad-CAM lab-vs-field figure.

Identical computation to 16_gradcam.py -- same FP32 tomato model, same
block_13_expand_relu layer, same seeded case selection, so the four cases
shown are the same four cases. What changes is presentation: the embedded
suptitle comes off (the caption carries it), typography joins the shared
figure system, and the correct/wrong distinction is carried by a coloured
frame as well as a glyph.

Story, left to right: the model has genuinely learned a lesion feature in
the lab (columns 1 and 3), and in the field the same attention machinery
fires on something that is not a lesion and lands on the attractor class
anyway (columns 2 and 4).
"""
import csv, json, re
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
import figstyle as fs

fs.apply()
IMG_SIZE = 96
LAYER_NAME = "block_13_expand_relu"

STOCK_PATTERNS = [
    re.compile(r"-[a-z0-9]{6,7}\.jpe?g$", re.I),
    re.compile(r"260nw|shutterstock|picture-id|gettyimages|istock", re.I),
    re.compile(r"leaf-\d{6,9}\.jpe?g$", re.I),
]


def looks_like_stock(path):
    return any(p.search(path.split("/")[-1]) for p in STOCK_PATTERNS)


model = tf.keras.models.load_model("model_tomato_fp32.keras")
grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[model.get_layer(LAYER_NAME).output, model.output])

classmap = json.load(open("classmap_tomato.json"))
remap = {int(k): v for k, v in classmap["remap"].items()}
rows = [r for r in csv.DictReader(open("manifest.csv")) if r["is_tomato"] == "True"]
idx_to_name = {int(r["class_idx"]): r["class_name"].split("___")[-1].replace("_", " ")
               for r in rows}
name_by_remapped = {remap[k]: v for k, v in idx_to_name.items()}

EARLY_BLIGHT_CLASS, HEALTHY_CLASS = 14, 18
EB_R, H_R = remap[EARLY_BLIGHT_CLASS], remap[HEALTHY_CLASS]


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [IMG_SIZE, IMG_SIZE]).numpy().astype(np.float32)


def predict_fp32(a):
    return int(np.argmax(model.predict((a / 127.5 - 1.0)[None], verbose=0)[0]))


def gradcam(a, class_idx):
    x = ((a / 127.5 - 1.0)[None]).astype(np.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    hm = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    return (tf.maximum(hm, 0) / (tf.reduce_max(hm) + 1e-8)).numpy()


def overlay(a, hm, alpha=0.42):
    im = tf.keras.utils.array_to_img(np.expand_dims(np.uint8(255 * hm), -1))
    hr = np.array(im.resize((IMG_SIZE, IMG_SIZE)))
    jet = cm.get_cmap("jet")(hr / 255.0)[:, :, :3] * 255
    return (alpha * jet + (1 - alpha) * a.astype(np.uint8)).astype(np.uint8)


lab_rows = [r for r in rows if r["domain"] == "pv_test" and not looks_like_stock(r["path"])]
field_rows = [r for r in rows if r["domain"] == "pd_field" and not looks_like_stock(r["path"])]
rng = np.random.default_rng(3)


def find_case(pool, want_true=None, want_pred=None, exclude_true=None):
    pool = list(pool)
    rng.shuffle(pool)
    for r in pool:
        t = int(r["class_idx"])
        tr = remap[t]
        if want_true is not None and t != want_true:
            continue
        if exclude_true is not None and t == exclude_true:
            continue
        a = load_img(r["path"])
        p = predict_fp32(a)
        if want_pred is not None and p != want_pred:
            continue
        if want_true is not None and p != tr:
            continue
        return r, a, p, tr
    return None


cases = [
    (find_case(lab_rows, want_true=EARLY_BLIGHT_CLASS),
     "Lab · true: Early blight"),
    (find_case(field_rows, exclude_true=EARLY_BLIGHT_CLASS, want_pred=EB_R),
     "Field · true: {true}"),
    (find_case(lab_rows, want_true=HEALTHY_CLASS),
     "Lab · true: Healthy"),
    (find_case(field_rows, exclude_true=HEALTHY_CLASS, want_pred=H_R),
     "Field · true: {true}"),
]

cell = fs.FULL_W / 4
fig, axes = fs.figure(width=fs.FULL_W, height=2 * (cell + 0.36), nrows=2, ncols=4)

for col, (case, tpl) in enumerate(cases):
    r, a, pred_idx, true_r = case
    true_name = name_by_remapped[true_r]
    pred_name = name_by_remapped[pred_idx]
    correct = pred_idx == true_r
    colour = fs.AQUA if correct else fs.RED

    axes[0, col].imshow(a.astype(np.uint8))
    axes[0, col].set_title(tpl.format(true=true_name), fontsize=fs.FS_ANNOT,
                           fontweight="normal", color=fs.INK, pad=3)

    axes[1, col].imshow(overlay(a, gradcam(a, pred_idx)))
    mark = "✓" if correct else "✗"
    axes[1, col].set_title("pred: %s  %s" % (pred_name, mark),
                           fontsize=fs.FS_ANNOT, fontweight="normal",
                           color=colour, pad=3)
    for sp in axes[1, col].spines.values():
        sp.set_visible(True)
        sp.set_color(colour)
        sp.set_linewidth(1.6)

for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
for ax in axes[0]:
    for sp in ax.spines.values():
        sp.set_visible(False)

axes[0, 0].set_ylabel("Input", fontsize=fs.FS_ANNOT, color=fs.INK_2)
axes[1, 0].set_ylabel("Grad-CAM at\npredicted class", fontsize=fs.FS_ANNOT,
                      color=fs.INK_2, linespacing=1.3)

fig.subplots_adjust(left=0.055, right=0.998, top=0.90, bottom=0.005,
                    wspace=0.06, hspace=0.22)
fs.finish(fig, "figures/gradcam_field_v3.png", tight=False, dpi=300)

for (case, _), nm in zip(cases, ["lab-EB-correct", "field-wrong-to-EB",
                                 "lab-healthy-correct", "field-wrong-to-healthy"]):
    r, a, p, tr = case
    print(nm, r["path"], "true:", name_by_remapped[tr], "pred:", name_by_remapped[p])
