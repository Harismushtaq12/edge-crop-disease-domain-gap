"""
Mitigation experiment: does a *small* amount of real field-labeled data close
the lab-to-field gap documented in the main experiments?

Protocol (few-shot field fine-tuning), run per experiment:
  1. Take the PlantDoc ("field") images for this experiment's classes and,
     per class, split them three ways (never touching the original
     PlantVillage train/val/lab-test split, and never touching the manifest
     used for the paper's headline numbers):
       - ft_train : up to 15 images/class -- used for gradient updates
       - ft_val   : up to 5 images/class  -- used only for early stopping
       - ft_test  : everything else       -- held out, final reported number,
                    never used for training or early stopping
  2. Load the already-trained from-scratch FP32 model for this experiment.
  3. Continue training (low LR, few epochs) on ft_train only (no lab images
     mixed in, to isolate the effect of the small field sample itself).
  4. Compare ft_test accuracy BEFORE (original model) vs AFTER fine-tuning,
     and also re-check lab (PlantVillage) accuracy after fine-tuning to
     check for catastrophic forgetting.

Usage: python3 07_finetune_mitigation.py <experiment: tomato|multicrop> [arch]
"""
import csv, json, random, sys, time
import numpy as np
import tensorflow as tf

EXPERIMENT = sys.argv[1]
ARCH = sys.argv[2] if len(sys.argv) > 2 else "mobilenetv2"
TAG = EXPERIMENT if ARCH == "mobilenetv2" else f"{EXPERIMENT}_{ARCH}"
IMG_SIZE = 96
BATCH = 16
FT_EPOCHS = 20
SEED = 123
random.seed(SEED)
tf.random.set_seed(SEED)

rows = list(csv.DictReader(open("manifest.csv")))
if EXPERIMENT == "tomato":
    rows = [r for r in rows if r["is_tomato"] == "True"]

classmap = json.load(open(f"classmap_{TAG}.json"))
remap = {int(k): v for k, v in classmap["remap"].items()}
n_classes = len(classmap["class_ids"])

field_rows = [r for r in rows if r["domain"] == "pd_field"]
by_class = {}
for r in field_rows:
    by_class.setdefault(r["class_idx"], []).append(r)

ft_train, ft_val, ft_test = [], [], []
for cid, rs in by_class.items():
    rs = rs[:]
    random.shuffle(rs)
    ft_train += rs[:15]
    ft_val += rs[15:20]
    ft_test += rs[20:]

print(f"[{TAG}] field-finetune split: ft_train={len(ft_train)} ft_val={len(ft_val)} ft_test={len(ft_test)}")

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return (img / 127.5 - 1.0)

def make_ds(rs, training):
    paths = [r["path"] for r in rs]
    labels = [remap[int(r["class_idx"])] for r in rs]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, l: (load_img(p), l), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(512, seed=SEED)

        def augment(img, label):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.15)
            img = tf.clip_by_value(img, -1.0, 1.0)
            return img, label

        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

ft_train_ds = make_ds(ft_train, True)
ft_val_ds = make_ds(ft_val, False)
ft_test_ds = make_ds(ft_test, False)

# also re-check lab accuracy isn't destroyed
test_rows = [r for r in rows if r["domain"] == "pv_test"]
lab_ds = make_ds(test_rows, False)

print(f"Loading base model: model_{TAG}_fp32.keras")
model = tf.keras.models.load_model(f"model_{TAG}_fp32.keras")

print("BEFORE fine-tuning:")
before_ft_test = model.evaluate(ft_test_ds, verbose=0)
before_lab = model.evaluate(lab_ds, verbose=0)
print(f"  ft_test acc={before_ft_test[1]:.4f}  lab acc={before_lab[1]:.4f}")

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
]
t0 = time.time()
history = model.fit(ft_train_ds, validation_data=ft_val_ds, epochs=FT_EPOCHS, callbacks=callbacks, verbose=2)
ft_time = time.time() - t0

print("AFTER fine-tuning:")
after_ft_test = model.evaluate(ft_test_ds, verbose=0)
after_lab = model.evaluate(lab_ds, verbose=0)
print(f"  ft_test acc={after_ft_test[1]:.4f}  lab acc={after_lab[1]:.4f}")

model.save(f"model_{TAG}_finetuned_fp32.keras")

# per-class breakdown on ft_test, before vs after
def per_class_acc(m, rs):
    if not rs:
        return {}
    by_c = {}
    for r in rs:
        by_c.setdefault(r["class_idx"], []).append(r)
    out = {}
    for cid, crs in by_c.items():
        ds = make_ds(crs, False)
        _, acc = m.evaluate(ds, verbose=0)
        out[cid] = {"acc": float(acc), "n": len(crs), "class_name": crs[0]["class_name"]}
    return out

# reload original (unmodified) model for a fair per-class BEFORE comparison
orig_model = tf.keras.models.load_model(f"model_{TAG}_fp32.keras")
before_perclass = per_class_acc(orig_model, ft_test)
after_perclass = per_class_acc(model, ft_test)

results = {
    "experiment": EXPERIMENT, "architecture": ARCH, "tag": TAG,
    "ft_train_n": len(ft_train), "ft_val_n": len(ft_val), "ft_test_n": len(ft_test),
    "ft_epochs_run": len(history.history["loss"]), "ft_time_sec": ft_time,
    "before": {"ft_test_accuracy": float(before_ft_test[1]), "lab_accuracy": float(before_lab[1])},
    "after": {"ft_test_accuracy": float(after_ft_test[1]), "lab_accuracy": float(after_lab[1])},
    "before_perclass": before_perclass,
    "after_perclass": after_perclass,
}
with open(f"results_{TAG}_finetune_mitigation.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSUMMARY")
print(f"  Field (ft_test) accuracy: {before_ft_test[1]*100:.1f}% -> {after_ft_test[1]*100:.1f}%  "
      f"(delta {(after_ft_test[1]-before_ft_test[1])*100:+.1f} pp)")
print(f"  Lab accuracy:             {before_lab[1]*100:.1f}% -> {after_lab[1]*100:.1f}%  "
      f"(delta {(after_lab[1]-before_lab[1])*100:+.1f} pp)")
print("DONE", json.dumps(results, indent=2))
