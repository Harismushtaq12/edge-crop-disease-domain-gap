"""
224x224 resolution ablation -- the paper's single largest remaining
confound, promoted from Future Work to a real experiment.

The question: Section V-C/V-G show that on field images, Grad-CAM attention
is at or below the chance level set by the leaf's own bounding box, and
that a real ImageNet prior does not fix this. But every model in this paper
runs at 96x96, chosen for TinyML SRAM/latency budgets. A reviewer's obvious
objection is that 96x96 may simply destroy the lesion-scale evidence
(a 10-pixel bacterial spot in a 224px image is ~4px at 96px, and after five
stride-2 stages it is sub-pixel in the feature map Grad-CAM reads). If the
disease signal is not resolvable, the network has no choice but to key on
whatever coarse context survives -- which would make the attention finding
a *low-resolution* artifact rather than a *domain-shift* one.

This script isolates exactly that variable: the same architecture, the same
ImageNet-pretrained initialization family, the same tomato splits, the same
optimizer/schedule/augmentation -- at 224x224 instead of 96x96, using
d-li14's 224-native checkpoint (mobilenetv2_1.0-0c6065bc.pth, 72.8% top-1,
the repo's headline model) rather than upsampling the 96x96 one, so the
prior is resolution-matched too.

Two changes are forced by resolution and are disclosed rather than hidden:
  * BATCH 64 -> 16. This is forced, and measured rather than assumed:
    47_batch_memory_probe.py sweeps batch size on the identical pipeline and
    records peak RSS (8: 3.6GB, 16: 4.1GB, 32: 5.7GB, 64: OOM-killed), against
    an 8.0GB container. Batch 64 extrapolates to ~8.5GB and is killed both in
    the probe and in a real training attempt. Batch 32 fits in isolation but
    not alongside another job, which is what killed the first attempt. The LR
    schedule and the BatchNorm momentum (m=0.90) are left identical, so running
    statistics still track batch statistics at the smaller batch.
    Pass the batch size as argv[1] to reproduce any of these.
  * The Grad-CAM target layer (34_) reads the same *named* layer as at
    96x96, block_13_expand_relu6, whose feature map is 14x14 here vs 6x6 at
    96x96 -- same network depth and semantic level, higher spatial detail,
    which is precisely the thing under test.

Everything else is byte-identical to 29_pretrained_finetune.py.

Usage: python3 33_resolution_ablation_224.py
"""
import csv
import json
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

sys.path.insert(0, ".")
import importlib.util
spec = importlib.util.spec_from_file_location("cvt", "28_convert_pretrained_weights.py")
cvt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cvt)

IMG_SIZE = 224
BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 16   # see docstring
EPOCHS = 60         # identical to 29_
SEED = 42           # identical to 29_
PORTED_WEIGHTS = "mobilenetv2_imagenet_224x224_ported.weights.h5"

tf.random.set_seed(SEED)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

rows = list(csv.DictReader(open("manifest.csv")))
rows = [r for r in rows if r["is_tomato"] == "True"]
class_ids = sorted(set(int(r["class_idx"]) for r in rows))
remap = {old: new for new, old in enumerate(class_ids)}
n_classes = len(class_ids)
print(f"[tomato/224px] classes={n_classes} total_rows={len(rows)}")


def split(domain):
    return [r for r in rows if r["domain"] == domain]


train_rows, val_rows = split("pv_train"), split("pv_val")
test_rows, field_rows = split("pv_test"), split("pd_field")
print(f"  train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} field(PlantDoc)={len(field_rows)}")


def make_ds(rs, training):
    paths = [r["path"] for r in rs]
    labels = [remap[int(r["class_idx"])] for r in rs]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = img / 255.0
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
        return (img - MEAN) / STD, label

    ds = ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE), len(rs)


train_ds, n_train = make_ds(train_rows, True)
val_ds, n_val = make_ds(val_rows, False)
test_ds, n_test = make_ds(test_rows, False)
field_ds, n_field = make_ds(field_rows, False)

full_model, _ = cvt.build_keras_mobilenetv2(IMG_SIZE, num_classes=1000, with_top=True)
full_model.load_weights(PORTED_WEIGHTS)
print("Loaded 224-native ImageNet backbone (mobilenetv2_1.0, 72.8% top-1), exact architecture match.")

feat = full_model.get_layer("avgpool").output
x = layers.Dense(n_classes, activation="softmax", name="tomato_head")(feat)
model = Model(full_model.input, x, name="mobilenetv2_pretrained224_tomato")

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


print("Evaluating...")
pv_test = eval_ds(test_ds, "PlantVillage test (lab, in-domain)")
pd_field = eval_ds(field_ds, "PlantDoc (field, out-of-domain)")

model.save(f"model_tomato_pretrained224_b{BATCH}_fp32.keras")

# reference points measured earlier in this paper, for the ablation table
BASE96_LAB, BASE96_FIELD = 0.9723435044288635, 0.22177419066429138   # 29_ (96px pretrained)
SCRATCH96_LAB, SCRATCH96_FIELD = 0.7874817848205566, 0.12096773833036423  # 03_ (96px scratch)

results = {
    "experiment": "tomato",
    "variant": "imagenet_pretrained_224px",
    "backbone_source": "d-li14/mobilenetv2.pytorch mobilenetv2_1.0-0c6065bc.pth (72.8% ImageNet top-1, 224-native)",
    "alpha": 1.0,
    "img_size": IMG_SIZE,
    "batch_size": BATCH,
    "n_classes": n_classes,
    "n_train": n_train, "n_val": n_val, "n_test": n_test, "n_field": n_field,
    "train_time_sec": train_time,
    "epochs_run": len(history.history["loss"]),
    "params": int(model.count_params()),
    "pv_test": pv_test,
    "pd_field": pd_field,
    "reference_96px_pretrained": {"lab": BASE96_LAB, "field": BASE96_FIELD},
    "reference_96px_scratch": {"lab": SCRATCH96_LAB, "field": SCRATCH96_FIELD},
    "delta_vs_96px_pretrained": {
        "lab": pv_test["accuracy"] - BASE96_LAB,
        "field": pd_field["accuracy"] - BASE96_FIELD,
    },
    "delta_domain_pp": (pv_test["accuracy"] - pd_field["accuracy"]) * 100,
}
with open(f"results_tomato_pretrained224_b{BATCH}_fp32.json", "w") as f:
    json.dump(results, f, indent=2)
print("DONE", json.dumps(results, indent=2))
