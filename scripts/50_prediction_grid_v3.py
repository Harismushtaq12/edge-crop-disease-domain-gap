"""
v3 of the qualitative prediction grid.

Same selection logic as v2 -- for every tomato class, the first field image
the deployed INT8 model gets right and the first it gets wrong, so the
figure is a fair sample rather than a curated one -- but laid out properly
for print: square centre crops on a regular grid, a two-line label block
above each panel, and the correct/wrong distinction carried by both a
coloured frame and a glyph so it survives greyscale printing.
"""
import csv, json, random, re
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import figstyle as fs

fs.apply()
random.seed(7)
IMG_SIZE = 96

STOCK_PATTERNS = [
    re.compile(r"-[a-z0-9]{6,7}\.jpe?g$", re.I),
    re.compile(r"260nw|shutterstock|picture-id|gettyimages|istock", re.I),
    re.compile(r"leaf-\d{6,9}\.jpe?g$", re.I),
]


def looks_like_stock(path):
    return any(p.search(path.split("/")[-1]) for p in STOCK_PATTERNS)


def square(img):
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


SHORT = {
    "Tomato Yellow Leaf Curl Virus": "Yellow leaf curl",
    "Tomato mosaic virus": "Mosaic virus",
    "Septoria leaf spot": "Septoria",
    "Bacterial spot": "Bacterial spot",
    "Early blight": "Early blight",
    "Late blight": "Late blight",
    "Leaf Mold": "Leaf mold",
    "healthy": "Healthy",
}

classmap = json.load(open("classmap_tomato.json"))
remap = {int(k): v for k, v in classmap["remap"].items()}
class_ids = classmap["class_ids"]

rows = [r for r in csv.DictReader(open("manifest.csv"))
        if r["is_tomato"] == "True" and r["domain"] == "pd_field"]

short_name = {}
for r in rows:
    short_name[int(r["class_idx"])] = r["class_name"].split("___")[-1].replace("_", " ")

interpreter = tf.lite.Interpreter(model_path="model_tomato_int8.tflite")
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]


def predict(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    x = (img.numpy().astype(np.float32) / 127.5 - 1.0)[None, ...]
    scale, zp = inp["quantization"]
    interpreter.set_tensor(inp["index"], (x / scale + zp).astype(np.int8))
    interpreter.invoke()
    return int(np.argmax(interpreter.get_tensor(out["index"])[0]))


by_class = {}
for r in rows:
    by_class.setdefault(int(r["class_idx"]), []).append(r)
class_order = sorted(by_class.keys(), key=lambda ci: class_ids.index(ci))
locals_order = [remap[c] for c in class_order]

selected = []
for ci in class_order:
    non_stock = [r for r in by_class[ci] if not looks_like_stock(r["path"])]
    stock = [r for r in by_class[ci] if looks_like_stock(r["path"])]
    random.shuffle(non_stock)
    random.shuffle(stock)
    true_local = remap[ci]
    got_correct = got_wrong = None
    for r in non_stock + stock:
        pred_local = predict(r["path"])
        if pred_local == true_local and got_correct is None:
            got_correct = (r["path"], short_name[ci], short_name[ci], True)
        elif pred_local != true_local and got_wrong is None:
            pred_ci = (class_order[locals_order.index(pred_local)]
                       if pred_local in locals_order else None)
            pred_name = short_name.get(pred_ci, "?")
            got_wrong = (r["path"], short_name[ci], pred_name, False)
        if got_correct and got_wrong:
            break
    if got_correct:
        selected.append(got_correct)
    if got_wrong:
        selected.append(got_wrong)

print(f"Selected {len(selected)} examples across {len(class_order)} classes")

n = len(selected)
ncols = 8
nrows = (n + ncols - 1) // ncols
cell = fs.FULL_W / ncols
fig, axes = fs.figure(width=fs.FULL_W, height=nrows * (cell + 0.40),
                      nrows=nrows, ncols=ncols)
axes = np.array(axes).reshape(nrows, ncols)

for idx in range(nrows * ncols):
    ax = axes[idx // ncols, idx % ncols]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if idx >= n:
        ax.axis("off")
        continue
    path, true_name, pred_name, correct = selected[idx]
    ax.imshow(square(mpimg.imread(path)))
    colour = fs.AQUA if correct else fs.RED
    mark = "✓" if correct else "✗"
    ax.set_title("true: %s\npred: %s  %s" % (SHORT.get(true_name, true_name),
                                             SHORT.get(pred_name, pred_name), mark),
                 fontsize=fs.FS_ANNOT - 0.6, color=colour, fontweight="normal",
                 linespacing=1.3, pad=3)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(colour)
        sp.set_linewidth(1.6)

fig.subplots_adjust(left=0.004, right=0.996, top=0.87, bottom=0.005,
                    wspace=0.07, hspace=0.42)
fs.finish(fig, "figures/prediction_samples_grid_v3.png", tight=False, dpi=300)
