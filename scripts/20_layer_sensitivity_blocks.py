"""
Finer-grained version of 17_layer_sensitivity.py's finding, at the block
granularity an external review specifically asked for: Stem / Blocks 1-3 /
Blocks 4-7 / Blocks 8-11 / SE units / Head, instead of the coarser
SE-vs-non-SE and early/middle/late-third split already in the paper.
Same fake-quantization method (per-tensor affine round-trip through int8,
Eq. 4), same tomato MobileNetV3-Small FP32 model, same lab-test accuracy
metric -- this is a re-grouping of the identical experiment, not a new one,
so it is directly comparable to Table results already reported.

MobileNetV3-Small's 11 inverted-residual blocks are `expanded_conv`
(block 1) through `expanded_conv_10` (block 11) inside the Keras
MobileNetV3Small backbone; `conv`/`conv_bn` is the stem, `conv_1`/
`conv_1_bn` plus the final `dense` classification layer are the head.
Blocks 2 and 3 (0-indexed 1, 2) are the only two blocks without a
squeeze-and-excite unit -- confirmed by inspecting layer names below.
"""
import csv, json
import numpy as np
import tensorflow as tf

IMG_SIZE = 96

def fake_quantize(w):
    lo, hi = w.min(), w.max()
    if hi <= lo:
        return w
    scale = (hi - lo) / 255.0
    zp = round(-lo / scale) - 128
    q = np.clip(np.round(w / scale) - zp, -128, 127)
    return (q + zp) * scale

model = tf.keras.models.load_model("model_tomato_mobilenetv3small_fp32.keras")
backbone = model.get_layer("MobileNetV3Small")
dense_head = model.get_layer("dense")

classmap = json.load(open("classmap_tomato_mobilenetv3small.json"))
remap = {int(k): v for k, v in classmap["remap"].items()}
rows = list(csv.DictReader(open("manifest.csv")))
rows = [r for r in rows if r["is_tomato"] == "True" and r["domain"] == "pv_test"]

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return (img.numpy().astype(np.float32) / 127.5) - 1.0

print(f"Loading {len(rows)} lab-test images...")
X = np.stack([load_img(r["path"]) for r in rows])
y_true = np.array([remap[int(r["class_idx"])] for r in rows])

def accuracy():
    preds = model.predict(X, batch_size=64, verbose=0)
    return (np.argmax(preds, axis=1) == y_true).mean()

baseline_acc = accuracy()
print(f"Baseline FP32 accuracy: {baseline_acc:.4f}")

all_layers = backbone.layers
block_prefixes = ["expanded_conv"] + [f"expanded_conv_{i}" for i in range(1, 11)]  # 11 blocks, 1..11

def layers_for_block(prefix):
    # exact-prefix match, not startswith, so "expanded_conv" doesn't also
    # grab "expanded_conv_1"'s layers
    out = []
    for l in all_layers:
        if len(l.get_weights()) == 0:
            continue
        rest = l.name[len(prefix):]
        if l.name == prefix or (l.name.startswith(prefix) and rest.startswith("_") and not rest[1:2].isdigit()):
            out.append(l)
    return out

block_layers = {i + 1: layers_for_block(p) for i, p in enumerate(block_prefixes)}
for i, layers in block_layers.items():
    assert len(layers) > 0, f"block {i} matched 0 layers"

stem_layers = [l for l in all_layers if l.name in ("conv", "conv_bn")]
head_layers = [l for l in all_layers if l.name in ("conv_1", "conv_1_bn")] + [dense_head]
se_layers = [l for l in all_layers if "squeeze_excite" in l.name]

groups = {
    "Stem": stem_layers,
    "Blocks 1-3": block_layers[1] + block_layers[2] + block_layers[3],
    "Blocks 4-7": block_layers[4] + block_layers[5] + block_layers[6] + block_layers[7],
    "Blocks 8-11": block_layers[8] + block_layers[9] + block_layers[10] + block_layers[11],
    "SE units": se_layers,
    "Head": head_layers,
}

def quantize_layers(layers):
    saved = {}
    for l in layers:
        w = l.get_weights()
        saved[l.name] = [x.copy() for x in w]
        newW = [fake_quantize(x) if x.ndim > 1 else x for x in w]
        l.set_weights(newW)
    return saved

def restore_layers(layers, saved):
    for l in layers:
        l.set_weights(saved[l.name])

results = {"baseline_fp32_acc": float(baseline_acc), "groups": {}}
for name, layers in groups.items():
    n_weighted = len(layers)
    n_params = sum(int(np.prod(w.shape)) for l in layers for w in l.get_weights() if w.ndim > 1)
    saved = quantize_layers(layers)
    acc = accuracy()
    restore_layers(layers, saved)
    drop = baseline_acc - acc
    results["groups"][name] = {
        "acc": float(acc), "drop": float(drop),
        "n_layers": n_weighted, "n_quantized_params": n_params,
    }
    print(f"  {name:14s} ({n_weighted:2d} layers, {n_params:>7,} params): "
          f"acc={acc:.4f}  drop={drop:+.4f}")

with open("layer_sensitivity_blocks_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved layer_sensitivity_blocks_results.json")
