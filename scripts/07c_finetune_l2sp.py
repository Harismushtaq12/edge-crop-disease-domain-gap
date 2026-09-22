"""
L2-SP regularized field fine-tuning: a third mitigation variant alongside
the paper's existing full-fine-tune and head-only-fine-tune (07_/07b_).

Motivation: Section V-E shows naive full fine-tuning trades 33.2 points of
lab accuracy for only 2.4 points of field-accuracy gain -- catastrophic
forgetting exactly as predicted by French 1999 and Yosinski et al. 2014.
L2-SP (Xuhong et al., "Explicit Inductive Bias for Transfer Learning with
Convolutional Networks", ICML 2018) was designed for precisely this
setting: adapting a source-trained model to a new domain with limited
target data, by penalizing the L2 distance of every backbone weight from
its *starting point* (the lab-trained model) instead of the origin, so the
model can move where the 15 field images/class actually push it, but pays
a cost for drifting far from what the lab data already taught it. Its
starting point here is naturally the already-trained FP32 lab model -- no
extra pretraining step needed.

Same ft_train (15/cls) / ft_val (5/cls) / ft_test (remainder) split as
07_/07b_ (identical seed=123), so results are directly comparable in
Table VII. Tries two regularization strengths (alpha) and reports the one
selected by ft_val accuracy only -- ft_test is never touched for model
selection, same discipline as the rest of the paper.

Usage: python3 07c_finetune_l2sp.py <experiment> [arch]
"""
import csv, json, random, sys, time
import numpy as np
import tensorflow as tf

EXPERIMENT = sys.argv[1] if len(sys.argv) > 1 else "tomato"
ARCH = sys.argv[2] if len(sys.argv) > 2 else "mobilenetv2"
TAG = EXPERIMENT if ARCH == "mobilenetv2" else f"{EXPERIMENT}_{ARCH}"
IMG_SIZE = 96
BATCH = 16
FT_EPOCHS = 20
SEED = 123
ALPHA_CANDIDATES = [1e-2, 1e-1]
random.seed(SEED)
tf.random.set_seed(SEED)

rows = list(csv.DictReader(open("manifest.csv")))
if EXPERIMENT == "tomato":
    rows = [r for r in rows if r["is_tomato"] == "True"]

classmap = json.load(open(f"classmap_{TAG}.json"))
remap = {int(k): v for k, v in classmap["remap"].items()}

field_rows = [r for r in rows if r["domain"] == "pd_field"]
by_class = {}
for r in field_rows:
    by_class.setdefault(r["class_idx"], []).append(r)

random.seed(SEED)  # match 07_/07b_'s split exactly (same seed, same shuffle order)
ft_train, ft_val, ft_test = [], [], []
for cid, rs in by_class.items():
    rs = rs[:]
    random.shuffle(rs)
    ft_train += rs[:15]
    ft_val += rs[15:20]
    ft_test += rs[20:]

print(f"[{TAG}] L2-SP field-finetune split: ft_train={len(ft_train)} ft_val={len(ft_val)} ft_test={len(ft_test)}")

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
test_rows = [r for r in rows if r["domain"] == "pv_test"]
lab_ds = make_ds(test_rows, False)

def eval_ds(model, ds):
    correct, total = 0, 0
    for x, y in ds:
        preds = model(x, training=False)
        pred_c = tf.argmax(preds, axis=1, output_type=tf.int32)
        correct += tf.reduce_sum(tf.cast(pred_c == y, tf.int32)).numpy()
        total += y.shape[0]
    return correct / total

def train_l2sp(alpha):
    print(f"\n=== L2-SP alpha={alpha} ===")
    model = tf.keras.models.load_model(f"model_{TAG}_fp32.keras")
    w0 = [tf.constant(v.numpy()) for v in model.trainable_variables]
    # exclude the final classifier layer's kernel+bias from L2-SP (standard
    # L2-SP design: penalize drift on inherited/backbone weights only, let
    # the task-specific head move freely). Keras 3 variable names are just
    # "kernel"/"bias" (no layer prefix), so identify the head by position:
    # the Dense classifier's kernel+bias are always the last two trainable
    # variables in this architecture (backbone, then head, in build order).
    n_vars = len(model.trainable_variables)
    anchor_mask = [i < n_vars - 2 for i in range(n_vars)]

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    best_val_acc = -1.0
    best_weights = None
    patience, bad_epochs = 6, 0

    for epoch in range(FT_EPOCHS):
        for x, y in ft_train_ds:
            with tf.GradientTape() as tape:
                preds = model(x, training=True)
                ce_loss = loss_fn(y, preds)
                l2sp_penalty = tf.add_n([
                    tf.reduce_sum(tf.square(v - w0_v))
                    for v, w0_v, keep in zip(model.trainable_variables, w0, anchor_mask) if keep
                ])
                loss = ce_loss + alpha * l2sp_penalty
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        val_acc = eval_ds(model, ft_val_ds)
        print(f"  epoch {epoch+1}: val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = [v.numpy().copy() for v in model.variables]
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop at epoch {epoch+1}")
                break

    for v, bw in zip(model.variables, best_weights):
        v.assign(bw)

    ft_test_acc = eval_ds(model, ft_test_ds)
    lab_acc = eval_ds(model, lab_ds)
    return model, best_val_acc, ft_test_acc, lab_acc

t0 = time.time()
candidates = []
for alpha in ALPHA_CANDIDATES:
    model, val_acc, ft_test_acc, lab_acc = train_l2sp(alpha)
    candidates.append((alpha, model, val_acc, ft_test_acc, lab_acc))
    print(f"  alpha={alpha}: val_acc={val_acc:.4f} ft_test_acc={ft_test_acc:.4f} lab_acc={lab_acc:.4f}")

# model selection by ft_val accuracy ONLY (never ft_test)
best = max(candidates, key=lambda c: c[2])
best_alpha, best_model, best_val_acc, best_ft_test_acc, best_lab_acc = best
total_time = time.time() - t0

orig_model = tf.keras.models.load_model(f"model_{TAG}_fp32.keras")
before_ft_test = eval_ds(orig_model, ft_test_ds)
before_lab = eval_ds(orig_model, lab_ds)

best_model.save(f"model_{TAG}_l2sp_fp32.keras")

results = {
    "experiment": EXPERIMENT, "architecture": ARCH, "tag": TAG, "method": "l2sp",
    "alpha_candidates": ALPHA_CANDIDATES, "alpha_selected": best_alpha,
    "selection_criterion": "ft_val_accuracy (ft_test never used for model selection)",
    "ft_train_n": len(ft_train), "ft_val_n": len(ft_val), "ft_test_n": len(ft_test),
    "total_time_sec": total_time,
    "before": {"ft_test_accuracy": float(before_ft_test), "lab_accuracy": float(before_lab)},
    "after": {"ft_test_accuracy": float(best_ft_test_acc), "lab_accuracy": float(best_lab_acc),
              "ft_val_accuracy": float(best_val_acc)},
    "all_alpha_results": [
        {"alpha": a, "ft_val_acc": float(v), "ft_test_acc": float(ft), "lab_acc": float(la)}
        for a, _, v, ft, la in candidates
    ],
}
with open(f"results_{TAG}_l2sp_mitigation.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSUMMARY (L2-SP, alpha=%s selected by ft_val)" % best_alpha)
print(f"  Field (ft_test) accuracy: {before_ft_test*100:.1f}% -> {best_ft_test_acc*100:.1f}%  "
      f"(delta {(best_ft_test_acc-before_ft_test)*100:+.1f} pp)")
print(f"  Lab accuracy:             {before_lab*100:.1f}% -> {best_lab_acc*100:.1f}%  "
      f"(delta {(best_lab_acc-before_lab)*100:+.1f} pp)")
print("DONE", json.dumps(results, indent=2))
