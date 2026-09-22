"""
Train the MobileNetV2 architecture (Sandler et al., 2018) FROM SCRATCH
(weights=None) on PlantVillage lab images, for the 21-class multi-crop set.

Note on methodology: this sandboxed research environment's network policy
blocks the hosts that normally serve ImageNet-pretrained Keras/TensorFlow
weights (storage.googleapis.com, download.pytorch.org, huggingface.co, and
GitHub release/LFS assets are all unreachable; only PyPI, npm, and plain git
clone / raw.githubusercontent.com work). Rather than fabricate a transfer-
learning result we cannot actually produce here, we train the same
architecture from scratch and report that honestly. This is disclosed in the
paper as a methodology note, not hidden.
"""
import csv, time, json, sys
import numpy as np
import tensorflow as tf

IMG_SIZE = 96
BATCH = 64
EPOCHS = 60
SEED = 42
EXPERIMENT = sys.argv[1] if len(sys.argv) > 1 else "multicrop"  # "multicrop" or "tomato"

tf.random.set_seed(SEED)

rows = list(csv.DictReader(open("manifest.csv")))
if EXPERIMENT == "tomato":
    rows = [r for r in rows if r["is_tomato"] == "True"]

# remap class indices to be contiguous for this experiment's subset
class_ids = sorted(set(int(r["class_idx"]) for r in rows))
remap = {old: new for new, old in enumerate(class_ids)}
n_classes = len(class_ids)
print(f"[{EXPERIMENT}] classes={n_classes}  total_rows={len(rows)}")

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
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds, len(rs)

train_ds, n_train = make_ds(train_rows, True)
val_ds, n_val = make_ds(val_rows, False)
test_ds, n_test = make_ds(test_rows, False)
field_ds, n_field = make_ds(field_rows, False)

model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), alpha=0.75, weights=None,
    classes=n_classes, classifier_activation="softmax",
)
# Fix: default BatchNorm momentum (0.999) needs far more steps than a short
# CPU-bound from-scratch run provides to converge its running statistics,
# which otherwise causes eval-mode predictions to collapse to a single
# constant class (train accuracy rises while val accuracy freezes at the
# majority-class baseline). Lowering momentum lets running stats track the
# true batch statistics within the epoch budget we actually have.
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.momentum = 0.90

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss="sparse_categorical_crossentropy",
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

model.save(f"model_{EXPERIMENT}_fp32.keras")
with open(f"results_{EXPERIMENT}_fp32.json", "w") as f:
    json.dump(results, f, indent=2)

# save class remap for later analysis
with open(f"classmap_{EXPERIMENT}.json", "w") as f:
    json.dump({"class_ids": class_ids, "remap": remap}, f, indent=2)

print("DONE", json.dumps(results, indent=2))
