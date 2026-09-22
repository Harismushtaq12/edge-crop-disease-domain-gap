"""
Generalized version of 03_train.py that supports multiple architectures,
for the multi-architecture comparison extension. Same from-scratch
constraint and same BN-collapse fix as the original (see 03_train.py's
docstring/comments for why).

Usage: python3 03b_train_arch.py <experiment: tomato|multicrop> <arch: mobilenetv3small|efficientnetb0>
"""
import csv, time, json, sys
import tensorflow as tf

IMG_SIZE = 96
BATCH = 64
EPOCHS = 60
SEED = 42
EXPERIMENT = sys.argv[1]
ARCH = sys.argv[2]
TAG = f"{EXPERIMENT}_{ARCH}"

tf.random.set_seed(SEED)

rows = list(csv.DictReader(open("manifest.csv")))
if EXPERIMENT == "tomato":
    rows = [r for r in rows if r["is_tomato"] == "True"]

class_ids = sorted(set(int(r["class_idx"]) for r in rows))
remap = {old: new for new, old in enumerate(class_ids)}
n_classes = len(class_ids)
print(f"[{TAG}] classes={n_classes}  total_rows={len(rows)}")

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
        img = img / 127.5 - 1.0
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

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
if ARCH == "mobilenetv2":
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), alpha=0.75, weights=None, include_top=False, pooling="avg",
    )
elif ARCH == "mobilenetv3small":
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), weights=None, include_top=False, pooling="avg",
        minimalistic=False,
    )
elif ARCH == "efficientnetb0":
    base = tf.keras.applications.EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), weights=None, include_top=False, pooling="avg",
    )
else:
    raise ValueError(f"unknown arch {ARCH}")

for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.momentum = 0.90

x = base(inputs)
outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

# Universal comparison-suite recipe: identical across all architectures so the
# multi-architecture table is a fair, apples-to-apples comparison. Peak LR is
# lower than the single-architecture MobileNetV2 run reported in the
# conference paper (3e-4) because MobileNetV3Small showed val_loss spikes
# (occasional BN/activation instability) at that LR during development;
# gradient clipping is added for the same reason. This is a *separate*,
# consistently-trained comparison suite -- it does not change the numbers
# already reported for the standalone MobileNetV2 experiments.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
print(model.count_params(), "parameters")

def lr_warmup(epoch, lr):
    if epoch == 0:
        return 3e-5
    if epoch == 1:
        return 6e-5
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
    "experiment": EXPERIMENT, "architecture": ARCH, "n_classes": n_classes, "img_size": IMG_SIZE,
    "n_train": n_train, "n_val": n_val, "n_test": n_test, "n_field": n_field,
    "train_time_sec": train_time, "epochs_run": len(history.history["loss"]),
    "params": int(model.count_params()),
}
print("Evaluating...")
results["pv_test"] = eval_ds(test_ds, "PlantVillage test (lab, in-domain)")
results["pd_field"] = eval_ds(field_ds, "PlantDoc (field, out-of-domain)")

model.save(f"model_{TAG}_fp32.keras")
with open(f"results_{TAG}_fp32.json", "w") as f:
    json.dump(results, f, indent=2)
with open(f"classmap_{TAG}.json", "w") as f:
    json.dump({"class_ids": class_ids, "remap": remap}, f, indent=2)

print("DONE", json.dumps(results, indent=2))
