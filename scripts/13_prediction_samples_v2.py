"""
v2 of the qualitative prediction-samples grid: same selection logic as
12_prediction_samples.py, but laid out as a WIDE landscape grid (8 columns x
2 rows) instead of a tall 4x4 grid, so it reads as a compact hero figure
inside an IEEE two-column layout instead of taking most of a page.
"""
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

STOCK_PATTERNS = [
    re.compile(r"-[a-z0-9]{6,7}\.jpe?g$", re.I),
    re.compile(r"260nw|shutterstock|picture-id|gettyimages|istock", re.I),
    re.compile(r"leaf-\d{6,9}\.jpe?g$", re.I),
]
def looks_like_stock(path):
    fname = path.split("/")[-1]
    return any(p.search(fname) for p in STOCK_PATTERNS)

classmap = json.load(open("classmap_tomato.json"))
remap = {int(k): v for k, v in classmap["remap"].items()}
class_ids = classmap["class_ids"]

rows = list(csv.DictReader(open("manifest.csv")))
rows = [r for r in rows if r["is_tomato"] == "True" and r["domain"] == "pd_field"]

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
    return int(np.argmax(logits))

by_class = {}
for r in rows:
    by_class.setdefault(int(r["class_idx"]), []).append(r)

class_order = sorted(by_class.keys(), key=lambda ci: class_ids.index(ci))

random.seed(7)
selected = []
for ci in class_order:
    non_stock = [r for r in by_class[ci] if not looks_like_stock(r["path"])]
    stock = [r for r in by_class[ci] if looks_like_stock(r["path"])]
    random.shuffle(non_stock); random.shuffle(stock)
    cand = non_stock + stock
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

# Wide landscape grid: 8 columns, as many rows as needed (16 examples -> 2 rows)
n = len(selected)
ncols = 8
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.62, nrows * 2.05))
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
    title = f"T: {true_name}\nP: {pred_name} {mark}"
    ax.set_title(title, fontsize=6.6, color=color, linespacing=1.25, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color(color); spine.set_linewidth(2.0)
    ax.set_xticks([]); ax.set_yticks([])

fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.01, wspace=0.08, hspace=0.55)
fig.savefig("figures/prediction_samples_grid.png", dpi=200)
from PIL import Image
print("saved", Image.open("figures/prediction_samples_grid.png").size)
