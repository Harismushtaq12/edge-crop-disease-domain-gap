"""
Stacked mitigation: ImageNet-pretrained backbone (29_) + few-shot field
fine-tuning (07b_/07c_), the combination the paper's own Future Work item
(1) names and a reviewer explicitly asked for.

Section V-E showed post-hoc few-shot fine-tuning of the FROM-SCRATCH model
moves field accuracy +2.4 to +5.6 pp at a 15.9-45.3 pp lab cost. Section
V-G showed the ImageNet-pretrained backbone alone moves field accuracy
+10.1 pp with zero field data. The obvious unanswered question is what
happens when you do both -- i.e. what the actual accuracy ceiling is with
all the data this paper has available.

This script applies the SAME two best-behaved fine-tuning variants from
Section IV-F -- head-only (frozen backbone) and L2-SP -- to the pretrained
backbone, with the IDENTICAL ft_train/ft_val/ft_test split (seed=123,
15/5/remainder per class) used by 07_/07b_/07c_, so the new rows drop
directly into Table VII/IX alongside the existing ones.

Two deliberate protocol carry-overs from 29_pretrained_finetune.py:
  * pixel normalization is the pretrained checkpoint's torchvision-style
    mean/std, NOT this paper's usual x/127.5-1 (the backbone's BatchNorm
    running statistics were calibrated against the former);
  * augmentation is applied in [0,1] space and normalization happens last.
Everything else -- optimizer, learning rates, patience, epoch cap, alpha
candidates, and the "select on ft_val only, never ft_test" discipline --
is copied unchanged from 07b_/07c_.

Usage: python3 32_stacked_pretrained_finetune.py
"""
import csv, json, random, time
import numpy as np
import tensorflow as tf

IMG_SIZE = 96
BATCH = 16
FT_EPOCHS = 20
SEED = 123
ALPHA_CANDIDATES = [1e-2, 1e-1]
HEAD_LR = 5e-4   # same as 07b_ (head-only updates far fewer params)
L2SP_LR = 5e-5   # same as 07c_
PRETRAINED_MODEL = "model_tomato_pretrained_fp32.keras"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

random.seed(SEED)
tf.random.set_seed(SEED)

rows = [r for r in csv.DictReader(open("manifest.csv")) if r["is_tomato"] == "True"]
classmap = json.load(open("classmap_tomato.json"))
remap = {int(k): v for k, v in classmap["remap"].items()}

field_rows = [r for r in rows if r["domain"] == "pd_field"]
by_class = {}
for r in field_rows:
    by_class.setdefault(r["class_idx"], []).append(r)

random.seed(SEED)  # identical split to 07_/07b_/07c_
ft_train, ft_val, ft_test = [], [], []
for cid, rs in by_class.items():
    rs = rs[:]
    random.shuffle(rs)
    ft_train += rs[:15]
    ft_val += rs[15:20]
    ft_test += rs[20:]

print(f"[stacked] split: ft_train={len(ft_train)} ft_val={len(ft_val)} ft_test={len(ft_test)}")


def make_ds(rs, training):
    paths = [r["path"] for r in rs]
    labels = [remap[int(r["class_idx"])] for r in rs]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_img(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        return img / 255.0, label

    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(512, seed=SEED)

        def augment(img, label):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.15 / 2.0)  # 0.15 in [-1,1] space == 0.075 in [0,1]
            img = tf.clip_by_value(img, 0.0, 1.0)
            return img, label

        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    def normalize(img, label):
        return (img - MEAN) / STD, label

    ds = ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


ft_train_ds = make_ds(ft_train, True)
ft_val_ds = make_ds(ft_val, False)
ft_test_ds = make_ds(ft_test, False)
lab_ds = make_ds([r for r in rows if r["domain"] == "pv_test"], False)
full_field_ds = make_ds(field_rows, False)  # all 744, for comparability with Table IX


def eval_ds(model, ds):
    correct, total = 0, 0
    for x, y in ds:
        preds = model(x, training=False)
        pred_c = tf.argmax(preds, axis=1, output_type=tf.int32)
        correct += tf.reduce_sum(tf.cast(pred_c == y, tf.int32)).numpy()
        total += y.shape[0]
    return correct / total


def train_head_only():
    """Freeze the pretrained backbone, train only the 8-class head."""
    print("\n=== stacked: pretrained + head-only fine-tune ===")
    model = tf.keras.models.load_model(PRETRAINED_MODEL)
    for layer in model.layers:
        layer.trainable = (layer.name == "tomato_head")
    n_trainable = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    print(f"  trainable params: {n_trainable}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=HEAD_LR),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"],
    )
    best_val, best_weights, bad = -1.0, None, 0
    for epoch in range(FT_EPOCHS):
        model.fit(ft_train_ds, epochs=1, verbose=0)
        val_acc = eval_ds(model, ft_val_ds)
        print(f"  epoch {epoch+1}: val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val, bad = val_acc, 0
            best_weights = [v.numpy().copy() for v in model.variables]
        else:
            bad += 1
            if bad >= 6:
                print(f"  early stop at epoch {epoch+1}")
                break
    for v, bw in zip(model.variables, best_weights):
        v.assign(bw)
    return model, best_val


def train_l2sp(alpha):
    """L2-SP: anchor backbone weights to the pretrained+lab-trained start."""
    print(f"\n=== stacked: pretrained + L2-SP alpha={alpha} ===")
    model = tf.keras.models.load_model(PRETRAINED_MODEL)
    w0 = [tf.constant(v.numpy()) for v in model.trainable_variables]
    n_vars = len(model.trainable_variables)
    anchor_mask = [i < n_vars - 2 for i in range(n_vars)]  # head (last 2) moves freely

    optimizer = tf.keras.optimizers.Adam(learning_rate=L2SP_LR)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    best_val, best_weights, bad = -1.0, None, 0

    for epoch in range(FT_EPOCHS):
        for x, y in ft_train_ds:
            with tf.GradientTape() as tape:
                preds = model(x, training=True)
                ce_loss = loss_fn(y, preds)
                penalty = tf.add_n([
                    tf.reduce_sum(tf.square(v - w0_v))
                    for v, w0_v, keep in zip(model.trainable_variables, w0, anchor_mask) if keep
                ])
                loss = ce_loss + alpha * penalty
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        val_acc = eval_ds(model, ft_val_ds)
        print(f"  epoch {epoch+1}: val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val, bad = val_acc, 0
            best_weights = [v.numpy().copy() for v in model.variables]
        else:
            bad += 1
            if bad >= 6:
                print(f"  early stop at epoch {epoch+1}")
                break
    for v, bw in zip(model.variables, best_weights):
        v.assign(bw)
    return model, best_val


t0 = time.time()

# --- before numbers: the pretrained model as Section V-G left it ---
base = tf.keras.models.load_model(PRETRAINED_MODEL)
before_ft_test = eval_ds(base, ft_test_ds)
before_lab = eval_ds(base, lab_ds)
before_full_field = eval_ds(base, full_field_ds)
print(f"\nBEFORE (pretrained, no field data): ft_test={before_ft_test:.4f} "
      f"lab={before_lab:.4f} full_field={before_full_field:.4f}")

variants = {}

head_model, head_val = train_head_only()
variants["head_only"] = {
    "ft_val_accuracy": float(head_val),
    "ft_test_accuracy": float(eval_ds(head_model, ft_test_ds)),
    "lab_accuracy": float(eval_ds(head_model, lab_ds)),
    "full_field_accuracy": float(eval_ds(head_model, full_field_ds)),
}
head_model.save("model_tomato_pretrained_headft_fp32.keras")
print("  head-only:", variants["head_only"])

l2sp_candidates = []
for alpha in ALPHA_CANDIDATES:
    m, v = train_l2sp(alpha)
    ft_test_acc = eval_ds(m, ft_test_ds)
    lab_acc = eval_ds(m, lab_ds)
    full_field_acc = eval_ds(m, full_field_ds)
    l2sp_candidates.append((alpha, m, v, ft_test_acc, lab_acc, full_field_acc))
    print(f"  alpha={alpha}: val={v:.4f} ft_test={ft_test_acc:.4f} lab={lab_acc:.4f} field={full_field_acc:.4f}")

# selection on ft_val ONLY
best_alpha, best_model, best_val, best_ft_test, best_lab, best_full_field = max(
    l2sp_candidates, key=lambda c: c[2])
variants["l2sp"] = {
    "alpha_selected": best_alpha,
    "ft_val_accuracy": float(best_val),
    "ft_test_accuracy": float(best_ft_test),
    "lab_accuracy": float(best_lab),
    "full_field_accuracy": float(best_full_field),
    "all_alpha_results": [
        {"alpha": a, "ft_val_acc": float(v), "ft_test_acc": float(ft),
         "lab_acc": float(la), "full_field_acc": float(ff)}
        for a, _, v, ft, la, ff in l2sp_candidates
    ],
}
best_model.save("model_tomato_pretrained_l2sp_fp32.keras")

results = {
    "experiment": "tomato",
    "variant": "stacked_pretrained_plus_fieldfinetune",
    "base_model": PRETRAINED_MODEL,
    "selection_criterion": "ft_val_accuracy (ft_test never used for model selection)",
    "ft_train_n": len(ft_train), "ft_val_n": len(ft_val), "ft_test_n": len(ft_test),
    "total_time_sec": time.time() - t0,
    "before": {
        "ft_test_accuracy": float(before_ft_test),
        "lab_accuracy": float(before_lab),
        "full_field_accuracy": float(before_full_field),
    },
    "after": variants,
    "reference_from_scratch_baseline": {
        "note": "Section V-E numbers, from-scratch backbone, same split",
        "headft_ft_test": 0.156, "headft_lab": 0.629,
        "l2sp_ft_test": 0.178, "l2sp_lab": 0.335,
    },
}
with open("results_tomato_stacked_mitigation.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSUMMARY (stacked: ImageNet-pretrained + few-shot field fine-tuning)")
print(f"  before          : ft_test={before_ft_test*100:.1f}%  lab={before_lab*100:.1f}%")
for name, v in variants.items():
    print(f"  {name:15s}: ft_test={v['ft_test_accuracy']*100:.1f}%  lab={v['lab_accuracy']*100:.1f}%  "
          f"(full-field {v['full_field_accuracy']*100:.1f}%)")
print("DONE", json.dumps(results, indent=2))
