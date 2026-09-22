"""
Real CutMix (Yun et al., ICCV 2019) training-time augmentation experiment,
added in response to an external review's "mitigations tested are a
strawman" critique: Section V-C shows the model's attention is only weakly
leaf-directed and shifts to background clutter on field images, and the
existing mitigations (Section IV-F/V-E) only intervene AFTER training via
field fine-tuning. This script tests the more obvious fix directly: does
forcing the model to see occluded/patched leaves DURING initial training
(instead of only afterward) reduce background reliance and narrow the
lab-to-field gap on its own, with zero field-labeled data?

Same architecture, same from-scratch recipe, same data splits as the
Section IV-B tomato baseline (03_train.py) -- the ONLY change is that
CutMix is applied to every training batch. This isolates CutMix's effect
from every other variable already reported in the paper.

CutMix: for each training batch, take a second, randomly shuffled copy of
the same batch; sample lambda ~ Beta(1,1); cut a lambda-proportional random
rectangular region out of the shuffled copy and paste it into the same
spot on the original image; mix the two images' one-hot labels by the
actual pasted-area fraction. This is implemented directly against the
tf.data batch (no tensorflow-probability dependency): Beta(a,a) is drawn
via two Gamma(a,1) samples, both natively supported by tf.random.gamma.
"""
import csv, time, json, sys
import numpy as np
import tensorflow as tf

IMG_SIZE = 96
BATCH = 64
EPOCHS = 60
SEED = 42
EXPERIMENT = "tomato"
CUTMIX_ALPHA = 1.0  # Beta(1,1) = uniform on [0,1], the CutMix paper's own default

tf.random.set_seed(SEED)
np.random.seed(SEED)

rows = list(csv.DictReader(open("manifest.csv")))
rows = [r for r in rows if r["is_tomato"] == "True"]

class_ids = sorted(set(int(r["class_idx"]) for r in rows))
remap = {old: new for new, old in enumerate(class_ids)}
n_classes = len(class_ids)
print(f"[{EXPERIMENT}-cutmix] classes={n_classes}  total_rows={len(rows)}")

def split(domain):
    return [r for r in rows if r["domain"] == domain]

train_rows = split("pv_train")
val_rows = split("pv_val")
test_rows = split("pv_test")
field_rows = split("pd_field")
print(f"  train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} field(PlantDoc)={len(field_rows)}")

def make_ds(rs, training):
    paths = [r["path"] for r in rs]
    labels = [remap[int(r["class_idx"])] for r in rs]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = img / 127.5 - 1.0  # match MobileNetV2 preprocess_input scaling
        label = tf.one_hot(label, n_classes)
        return img, label

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2048, seed=SEED)

        def augment(img, label):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_contrast(img, 0.9, 1.1)
            img = tf.clip_by_value(img, -1.0, 1.0)
            return img, label

        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH, drop_remainder=training).prefetch(tf.data.AUTOTUNE)
    return ds, len(rs)

def sample_beta(alpha):
    g1 = tf.random.gamma([], alpha, 1.0)
    g2 = tf.random.gamma([], alpha, 1.0)
    return g1 / (g1 + g2)

def cutmix_batch(images, labels):
    batch_size = tf.shape(images)[0]
    perm = tf.random.shuffle(tf.range(batch_size))
    shuf_images = tf.gather(images, perm)
    shuf_labels = tf.gather(labels, perm)

    lam = sample_beta(CUTMIX_ALPHA)
    cut_ratio = tf.sqrt(1.0 - lam)
    cut_w = tf.cast(tf.cast(IMG_SIZE, tf.float32) * cut_ratio, tf.int32)
    cut_h = tf.cast(tf.cast(IMG_SIZE, tf.float32) * cut_ratio, tf.int32)
    cx = tf.random.uniform([], 0, IMG_SIZE, dtype=tf.int32)
    cy = tf.random.uniform([], 0, IMG_SIZE, dtype=tf.int32)
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, IMG_SIZE)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, IMG_SIZE)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, IMG_SIZE)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, IMG_SIZE)

    # boolean mask, True = pixel comes from the shuffled ("pasted-in") image
    row_ids = tf.range(IMG_SIZE)[:, None]
    col_ids = tf.range(IMG_SIZE)[None, :]
    box_mask = (row_ids >= y1) & (row_ids < y2) & (col_ids >= x1) & (col_ids < x2)
    box_mask = tf.reshape(box_mask, [1, IMG_SIZE, IMG_SIZE, 1])
    box_mask = tf.tile(box_mask, [batch_size, 1, 1, 3])

    mixed_images = tf.where(box_mask, shuf_images, images)

    box_area = tf.cast((x2 - x1) * (y2 - y1), tf.float32)
    lam_adj = 1.0 - box_area / tf.cast(IMG_SIZE * IMG_SIZE, tf.float32)
    mixed_labels = lam_adj * labels + (1.0 - lam_adj) * shuf_labels
    return mixed_images, mixed_labels

train_ds, n_train = make_ds(train_rows, True)
train_ds = train_ds.map(cutmix_batch, num_parallel_calls=tf.data.AUTOTUNE)
val_ds, n_val = make_ds(val_rows, False)
test_ds, n_test = make_ds(test_rows, False)
field_ds, n_field = make_ds(field_rows, False)

model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), alpha=0.75, weights=None,
    classes=n_classes, classifier_activation="softmax",
)
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.momentum = 0.90

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss="categorical_crossentropy",  # soft (mixed) labels require this, not sparse CE
    metrics=["accuracy"],
)
print(model.count_params(), "parameters")

def lr_warmup(epoch, lr):
    if epoch == 0:
        return 1e-4
    if epoch == 1:
        return 2e-4
    return lr

callbacks = [
    tf.keras.callbacks.LearningRateScheduler(lr_warmup),
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=5, min_lr=1e-5),
]

t0 = time.time()
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)
train_time = time.time() - t0
print(f"Training took {train_time:.0f}s")

def eval_ds(ds, name):
    loss, acc = model.evaluate(ds, verbose=0)
    print(f"  {name}: loss={loss:.4f} acc={acc:.4f}")
    return {"loss": float(loss), "accuracy": float(acc)}

results = {
    "experiment": EXPERIMENT,
    "variant": "cutmix_training",
    "cutmix_alpha": CUTMIX_ALPHA,
    "n_classes": n_classes,
    "img_size": IMG_SIZE,
    "n_train": n_train, "n_val": n_val, "n_test": n_test, "n_field": n_field,
    "train_time_sec": train_time,
    "epochs_run": len(history.history["loss"]),
    "params": int(model.count_params()),
}
print("Evaluating...")
results["pv_test"] = eval_ds(test_ds, "PlantVillage test (lab, in-domain)")
results["pd_field"] = eval_ds(field_ds, "PlantDoc (field, out-of-domain)")

# comparison against the non-CutMix baseline already in the paper
baseline = json.load(open("results_tomato_fp32.json"))
results["baseline_pv_test_acc"] = baseline["pv_test"]["accuracy"]
results["baseline_pd_field_acc"] = baseline["pd_field"]["accuracy"]
results["delta_lab_acc"] = results["pv_test"]["accuracy"] - baseline["pv_test"]["accuracy"]
results["delta_field_acc"] = results["pd_field"]["accuracy"] - baseline["pd_field"]["accuracy"]

model.save(f"model_{EXPERIMENT}_cutmix_fp32.keras")
with open(f"results_{EXPERIMENT}_cutmix_fp32.json", "w") as f:
    json.dump(results, f, indent=2)

print("DONE", json.dumps(results, indent=2))
