"""
Build a manifest of images for the Edge AI plant-disease experiment.

Sources:
  - PlantVillage (lab-condition images) -> train/val/test split, used for training
  - PlantDoc (real-world field-condition images) -> held out entirely, used only
    as an out-of-domain field-accuracy evaluation set

Only classes with a genuine PlantVillage <-> PlantDoc match are included, so
every class we train on also has a real field-condition test set.
"""
import os, csv, random

random.seed(42)

PV_ROOT = "/tmp/plantvillage_test/raw/color"
PD_ROOT = "/tmp/plantdoc_test"

# (PlantVillage folder, PlantDoc folder, crop, is_tomato)
CLASS_MAP = [
    ("Apple___Apple_scab", "Apple Scab Leaf", "Apple", False),
    ("Apple___Cedar_apple_rust", "Apple rust leaf", "Apple", False),
    ("Apple___healthy", "Apple leaf", "Apple", False),
    ("Corn_(maize)___Northern_Leaf_Blight", "Corn leaf blight", "Corn", False),
    ("Corn_(maize)___Common_rust_", "Corn rust leaf", "Corn", False),
    ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn Gray leaf spot", "Corn", False),
    ("Grape___Black_rot", "grape leaf black rot", "Grape", False),
    ("Grape___healthy", "grape leaf", "Grape", False),
    ("Pepper,_bell___Bacterial_spot", "Bell_pepper leaf spot", "Pepper", False),
    ("Pepper,_bell___healthy", "Bell_pepper leaf", "Pepper", False),
    ("Potato___Early_blight", "Potato leaf early blight", "Potato", False),
    ("Potato___Late_blight", "Potato leaf late blight", "Potato", False),
    ("Squash___Powdery_mildew", "Squash Powdery mildew leaf", "Squash", False),
    ("Tomato___Bacterial_spot", "Tomato leaf bacterial spot", "Tomato", True),
    ("Tomato___Early_blight", "Tomato Early blight leaf", "Tomato", True),
    ("Tomato___Late_blight", "Tomato leaf late blight", "Tomato", True),
    ("Tomato___Septoria_leaf_spot", "Tomato Septoria leaf spot", "Tomato", True),
    ("Tomato___Tomato_mosaic_virus", "Tomato leaf mosaic virus", "Tomato", True),
    ("Tomato___healthy", "Tomato leaf", "Tomato", True),
    ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato leaf yellow virus", "Tomato", True),
    ("Tomato___Leaf_Mold", "Tomato mold leaf", "Tomato", True),
]

CAP_PER_CLASS = 600  # cap PlantVillage images per class to keep CPU training tractable

rows = []
class_id_map = {}
for idx, (pv, pd, crop, is_tom) in enumerate(CLASS_MAP):
    class_id_map[pv] = idx

    pv_dir = os.path.join(PV_ROOT, pv)
    files = [f for f in os.listdir(pv_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(files)
    files = files[:CAP_PER_CLASS]
    n = len(files)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    for i, f in enumerate(files):
        split = "pv_train" if i < n_train else ("pv_val" if i < n_train + n_val else "pv_test")
        rows.append({
            "path": os.path.join(pv_dir, f),
            "class_idx": idx,
            "class_name": pv,
            "crop": crop,
            "is_tomato": is_tom,
            "domain": split,
        })

    # PlantDoc field images: combine test/ and train/ folders for this class (all used as OOD eval)
    pd_files = []
    for split_dir in ["test", "train"]:
        d = os.path.join(PD_ROOT, split_dir, pd)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    pd_files.append(os.path.join(d, f))
    for f in pd_files:
        rows.append({
            "path": f,
            "class_idx": idx,
            "class_name": pv,
            "crop": crop,
            "is_tomato": is_tom,
            "domain": "pd_field",
        })

# write manifest
out_path = "manifest.csv"
with open(out_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["path", "class_idx", "class_name", "crop", "is_tomato", "domain"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

# write class list
with open("classes.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["class_idx", "pv_class", "pd_class", "crop", "is_tomato"])
    for idx, (pv, pd, crop, is_tom) in enumerate(CLASS_MAP):
        w.writerow([idx, pv, pd, crop, is_tom])

# summary
from collections import Counter
dom_counts = Counter(r["domain"] for r in rows)
print("Total rows:", len(rows))
print("By domain:", dict(dom_counts))
per_class_pd = Counter((r["class_name"]) for r in rows if r["domain"] == "pd_field")
print("\nPlantDoc field images per class (min/max/mean):",
      min(per_class_pd.values()), max(per_class_pd.values()), sum(per_class_pd.values())/len(per_class_pd))
for pv, pd, crop, is_tom in CLASS_MAP:
    print(f"  {pv:55s} field_n={per_class_pd.get(pv, 0):4d}")
