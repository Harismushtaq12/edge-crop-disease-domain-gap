import csv, json, random
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import re
random.seed(7)
IMG_SIZE = 96

# Heuristic filter to avoid picking images that are visibly watermarked
# stock-photo previews (Alamy/Shutterstock/Dreamstime/Getty/iStock), which
# PlantDoc's internet-sourced set occasionally includes. Filenames from these
# services follow recognizable ID/slug patterns; skip them when an
# alternative exists so the figure only shows freely reusable field photos.
STOCK_PATTERNS = [
    re.compile(r"-[a-z0-9]{6,7}\.jpe?g$", re.I),   # Alamy-style trailing ID, e.g. -BH3TH0.jpg
    re.compile(r"260nw|shutterstock|picture-id|gettyimages|istock", re.I),
    re.compile(r"leaf-\d{6,9}\.jpe?g$", re.I),      # Dreamstime-style, e.g. tomato-leaf-6424287.jpg
]
def looks_like_stock(path):
    fname = path.split("/")[-1]
    return any(p.search(fname) for p in STOCK_PATTERNS)

classmap = json.load(open("classmap_tomato.json"))
remap = {int(k): v for k, v in classmap["remap"].items()}
class_ids = classmap["class_ids"]

rows = list(csv.DictReader(open("manifest.csv")))
rows = [r for r in rows if r["is_tomato"] == "True" and r["domain"] == "pd_field"]

# class_idx -> short display name (matches paper's Table IV naming)
short_name = {}
for r in rows:
    short_name[int(r["class_idx"])] = r["class_name"].split("___")[-1].replace("_", " ")

interpreter = tf.lite.Interpreter(model_path="model_tomato_int8.tflite")
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 127.5 - 1.0
    return img.numpy().astype(np.float32)

def predict(path):
    img = load_img(path)
    x = np.expand_dims(img, 0)
    scale, zero_point = inp["quantization"]
    x = (x / scale + zero_point).astype(np.int8)
    interpreter.set_tensor(inp["index"], x)
    interpreter.invoke()
    logits = interpreter.get_tensor(out["index"])[0]
    pred = int(np.argmax(logits))
    return pred

# group rows by class_idx
by_class = {}
for r in rows:
    by_class.setdefault(int(r["class_idx"]), []).append(r)

class_order = sorted(by_class.keys(), key=lambda ci: class_ids.index(ci))
disp_names = [short_name[ci] for ci in class_order]

# For each class: find one correct and one incorrect prediction (if available)
random.seed(7)
selected = []  # list of (path, true_name, pred_name, correct_bool)
for ci in class_order:
    non_stock = [r for r in by_class[ci] if not looks_like_stock(r["path"])]
    stock = [r for r in by_class[ci] if looks_like_stock(r["path"])]
    random.shuffle(non_stock); random.shuffle(stock)
    cand = non_stock + stock  # prefer non-stock candidates first
    true_local = remap[ci]
    got_correct, got_wrong = None, None
    for r in cand:
        pred_local = predict(r["path"])
        is_correct = (pred_local == true_local)
        if is_correct and got_correct is None:
            got_correct = (r["path"], short_name[ci], short_name[ci], True)
        if (not is_correct) and got_wrong is None:
            pred_ci = class_order[[remap[c] for c in class_order].index(pred_local)] if pred_local in [remap[c] for c in class_order] else None
            pred_name = short_name.get(pred_ci, "?") if pred_ci is not None else "?"
            got_wrong = (r["path"], short_name[ci], pred_name, False)
        if got_correct and got_wrong:
            break
    if got_correct: selected.append(got_correct)
    if got_wrong: selected.append(got_wrong)

print(f"Selected {len(selected)} examples across {len(class_order)} classes")
for s in selected:
    print(s[1], "->", s[2], "correct" if s[3] else "WRONG")

# Build grid: up to 16 images, 4 cols
n = len(selected)
ncols = 4
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2.65, nrows*3.05))
axes = np.array(axes).reshape(nrows, ncols)

for idx in range(nrows * ncols):
    ax = axes[idx // ncols, idx % ncols]
    ax.axis("off")
    if idx >= n:
        continue
    path, true_name, pred_name, correct = selected[idx]
    img = mpimg.imread(path)
    ax.imshow(img)
    color = "#1b7a1b" if correct else "#b3261e"
    mark = "✓" if correct else "✗"
    title = f"True: {true_name}\nPred: {pred_name}  {mark}"
    ax.set_title(title, fontsize=8.3, color=color, linespacing=1.35)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color(color); spine.set_linewidth(2.2)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("Qualitative predictions, tomato INT8 model, PlantDoc field images — one correct + one incorrect example per class (where available)", fontsize=10.5, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig("figures/prediction_samples_grid.png", dpi=180)
from PIL import Image
print("saved", Image.open("figures/prediction_samples_grid.png").size)
