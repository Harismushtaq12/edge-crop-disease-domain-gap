"""
Trains the tomato model with a REAL ImageNet-pretrained MobileNetV2 backbone
(ported in 28_convert_pretrained_weights.py from d-li14/mobilenetv2.pytorch's
mobilenetv2_96x96-ff0e83d8.pth, verified to cosine-sim=1.00000 against the
source PyTorch checkpoint) instead of random initialization -- the direct
test of the "tabula rasa confound" critique: is the 78.7%->12.1% lab-to-field
collapse a domain-shift finding, or just what any from-scratch network does
on ~3,200 training images?

Kept IDENTICAL to the from-scratch baseline (03_train.py) to isolate
initialization as the only manipulated variable: same tomato manifest rows/
splits/seed, same optimizer (Adam), same LR warmup/schedule, same epoch
budget and early-stopping/plateau callbacks, same batch size, same
augmentation operations, same BatchNorm momentum fix. Two things necessarily
differ, both forced by which checkpoint is actually obtainable rather than
chosen for convenience:
  1. Width multiplier: alpha=1.0, not the paper's usual alpha=0.75 -- no
     0.75-width, 96x96-native checkpoint exists in the source repo, and
     interpolating one would not be a real pretrained weight.
  2. Preprocessing: (pixel/255 - mean)/std (torchvision-style), not the
     baseline's pixel/127.5-1 (Keras-style) -- the ported backbone's
     BatchNorm running statistics were calibrated against the former during
     its original ImageNet training, so using the baseline's scaling would
     silently feed the pretrained backbone out-of-distribution inputs.
Both are disclosed, not hidden -- this is a resource-constrained-sandbox
experiment, not a strawman.
"""
import csv
import json
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

sys.path.insert(0, ".")
# Module name starts with a digit, so it can't be `import`ed normally --
# load 28_convert_pretrained_weights.py directly by file path instead.
import importlib.util
spec = importlib.util.spec_from_file_location("cvt", "28_convert_pretrained_weights.py")
cvt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cvt)

IMG_SIZE = 96
BATCH = 64
EPOCHS = 60
SEED = 42
PORTED_WEIGHTS = "mobilenetv2_imagenet_96x96_ported.weights.h5"

tf.random.set_seed(SEED)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---------------- same tomato manifest rows/splits as 03_train.py ----------------
rows = list(csv.DictReader(open("manifest.csv")))
rows = [r for r in rows if r["is_tomato"] == "True"]
class_ids = sorted(set(int(r["class_idx"]) for r in rows))
remap = {old: new for new, old in enumerate(class_ids)}
n_classes = len(class_ids)
print(f"[tomato/pretrained] classes={n_classes} total_rows={len(rows)}")


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
        img = img / 255.0  # [0,1], augmentation happens here, normalize last
        return img, label

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2048, seed=SEED)

        def augment(img, label):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_contrast(img, 0.9, 1.1)
            img = tf.clip_by_value(img, 0.0, 1.0)
            return img, label

        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    def normalize(img, label):
        img = (img - MEAN) / STD  # torchvision-style, matches pretrained backbone
        return img, label

    ds = ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds, len(rs)


train_ds, n_train = make_ds(train_rows, True)
val_ds, n_val = make_ds(val_rows, False)
test_ds, n_test = make_ds(test_rows, False)
field_ds, n_field = make_ds(field_rows, False)

# ---------------- pretrained backbone + fresh tomato head ----------------
# Keras 3's native `.weights.h5` format doesn't support `by_name` loading,
# so load the full (architecture-matched) 1000-class model first, then
# re-wrap its feature output (pre-classifier) with a fresh tomato head,
# rather than trying to load weights directly into a separately built
# feature-only model.
full_model, _ = cvt.build_keras_mobilenetv2(IMG_SIZE, num_classes=1000, with_top=True)  # alpha=1.0
full_model.load_weights(PORTED_WEIGHTS)
print("Loaded pretrained ImageNet backbone weights (full 1000-class model, exact architecture match).")

feat = full_model.get_layer("avgpool").output
x = layers.Dense(n_classes, activation="softmax", name="tomato_head")(feat)
model = Model(full_model.input, x, name="mobilenetv2_pretrained_tomato")

# same BatchNorm-momentum fix as the from-scratch baseline, applied uniformly
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.momentum = 0.90

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
print(model.count_params(), "parameters (alpha=1.0 backbone + fresh 8-class head)")


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
    "experiment": "tomato",
    "variant": "imagenet_pretrained",
    "backbone_source": "d-li14/mobilenetv2.pytorch mobilenetv2_96x96-ff0e83d8.pth (62.696% ImageNet top-1)",
    "alpha": 1.0,
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

baseline = json.load(open("results_tomato_fp32.json"))
results["baseline_pv_test_acc"] = baseline["pv_test"]["accuracy"]
results["baseline_pd_field_acc"] = baseline["pd_field"]["accuracy"]
results["delta_lab_acc"] = results["pv_test"]["accuracy"] - baseline["pv_test"]["accuracy"]
results["delta_field_acc"] = results["pd_field"]["accuracy"] - baseline["pd_field"]["accuracy"]

model.save("model_tomato_pretrained_fp32.keras")
with open("results_tomato_pretrained_fp32.json", "w") as f:
    json.dump(results, f, indent=2)

print("DONE", json.dumps(results, indent=2))
