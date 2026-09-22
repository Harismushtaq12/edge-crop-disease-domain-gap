"""
Unified scoreboard: every trained tomato variant, scored on one consistent
basis so the paper can plot them all in the same lab-vs-field plane.

Why this is needed. The variants in this paper were, historically, each
reported against whichever field set was appropriate for that experiment:
the zero-field-data arms (baseline, CutMix, ImageNet-pretrained) against
the full 744-image PlantDoc set, and the few-shot fine-tuning arms against
the 584-image D_ft-test split, because for those the other 160 images were
consumed by ft_train/ft_val. Both choices are individually right, but they
are not mutually comparable, and a reader cannot put all the variants on
one axis without a common denominator.

This script recomputes, for every saved checkpoint:
  * lab accuracy   (PlantVillage held-out test, 687 images)
  * field_full     (all 744 PlantDoc tomato images)
  * field_fttest   (the 584-image D_ft-test split only)
D_ft-test is the honest common basis: it is disjoint from ft_train/ft_val
for *every* variant, so no model has seen any of it, including the ones
that never used field data at all.

Each checkpoint is scored with the preprocessing it was actually trained
with -- x/127.5-1 for the from-scratch family, torchvision mean/std for the
ported-ImageNet family -- which is why PREPROC is declared per model rather
than assumed.

Usage: python3 37_unified_scoreboard.py
"""
import csv, json, random
import numpy as np
import tensorflow as tf

IMG_SIZE = 96
BATCH = 32
SEED = 123   # the split seed used by 07_/07b_/07c_/32_

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# model file -> (label, preprocessing family)
MODELS = [
    ("model_tomato_fp32.keras",                  "From-scratch baseline",      "scale"),
    ("model_tomato_finetuned_fp32.keras",        "+ full fine-tune",           "scale"),
    ("model_tomato_headft_fp32.keras",           "+ head-only fine-tune",      "scale"),
    ("model_tomato_l2sp_fp32.keras",             "+ L2-SP fine-tune",          "scale"),
    ("model_tomato_cutmix_fp32.keras",           "+ CutMix",                   "scale"),
    ("model_tomato_pretrained_fp32.keras",       "ImageNet-pretrained",        "torchvision"),
    ("model_tomato_pretrained_headft_fp32.keras","Pretrained + head-only",     "torchvision"),
    ("model_tomato_pretrained_l2sp_fp32.keras",  "Pretrained + L2-SP",         "torchvision"),
]

rows = [r for r in csv.DictReader(open("manifest.csv")) if r["is_tomato"] == "True"]
classmap = json.load(open("classmap_tomato.json"))
remap = {int(k): v for k, v in classmap["remap"].items()}

field_rows = [r for r in rows if r["domain"] == "pd_field"]
lab_rows = [r for r in rows if r["domain"] == "pv_test"]

by_class = {}
for r in field_rows:
    by_class.setdefault(r["class_idx"], []).append(r)
random.seed(SEED)
ft_train, ft_val, ft_test = [], [], []
for cid, rs in by_class.items():
    rs = rs[:]
    random.shuffle(rs)
    ft_train += rs[:15]
    ft_val += rs[15:20]
    ft_test += rs[20:]
print(f"splits: lab={len(lab_rows)} field_full={len(field_rows)} "
      f"field_fttest={len(ft_test)} (held out from every variant)")


def make_ds(rs, preproc):
    paths = [r["path"] for r in rs]
    labels = [remap[int(r["class_idx"])] for r in rs]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        if preproc == "scale":
            img = img / 127.5 - 1.0
        else:
            img = ((img / 255.0) - MEAN) / STD
        return img, label

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)


def accuracy(model, ds):
    correct = total = 0
    for x, y in ds:
        p = tf.argmax(model(x, training=False), axis=1, output_type=tf.int32)
        correct += int(tf.reduce_sum(tf.cast(p == y, tf.int32)).numpy())
        total += int(y.shape[0])
    return correct / total


out = []
for path, label, preproc in MODELS:
    model = tf.keras.models.load_model(path)
    rec = {
        "model": path, "label": label, "preproc": preproc,
        "lab": accuracy(model, make_ds(lab_rows, preproc)),
        "field_full": accuracy(model, make_ds(field_rows, preproc)),
        "field_fttest": accuracy(model, make_ds(ft_test, preproc)),
    }
    rec["delta_domain_fttest_pp"] = (rec["lab"] - rec["field_fttest"]) * 100
    rec["uses_field_data"] = "fine-tune" in label or "L2-SP" in label or "head-only" in label
    out.append(rec)
    print(f"{label:28s} lab={rec['lab']*100:5.1f}%  field_full={rec['field_full']*100:5.1f}%  "
          f"field_fttest={rec['field_fttest']*100:5.1f}%  Δ={rec['delta_domain_fttest_pp']:5.1f}pp")
    del model
    tf.keras.backend.clear_session()

with open("unified_scoreboard.json", "w") as f:
    json.dump({"img_size": IMG_SIZE, "split_seed": SEED,
               "n_lab": len(lab_rows), "n_field_full": len(field_rows),
               "n_field_fttest": len(ft_test), "variants": out}, f, indent=2)
print("DONE -> unified_scoreboard.json")
