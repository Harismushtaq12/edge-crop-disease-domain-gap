"""
Per-layer / per-block INT8 weight-quantization sensitivity analysis for
MobileNetV3-Small (tomato model) -- diagnoses *which* layers drive the real
57.1% -> 13.1% INT8 collapse reported in the paper (Table V/VI), resolving
the paper's own disclosed "Undiagnosed quantization failure" limitation.

Method: fake-quantize (round-trip through int8 and back to float, per-
tensor affine, matching the paper's own Eq. 4: scale=(max-min)/255, clip to
[-128,127]) the *weights* of one layer or group of layers at a time, leave
every other layer at full FP32 precision, and measure lab-test accuracy.
This isolates each layer/group's individual contribution to the collapse
without needing a full TFLite re-export per configuration. It is a
simplification of TFLite's real per-channel, weights+activations INT8
scheme (disclosed as such in the paper) but is the standard method used in
the mixed-precision-quantization literature for exactly this kind of
layer-sensitivity ranking.
"""
import csv, json
import numpy as np
import tensorflow as tf

IMG_SIZE = 96

def fake_quantize(w):
    """Round-trip w through per-tensor affine int8 quantization (Eq. 4)."""
    lo, hi = w.min(), w.max()
    if hi <= lo:
        return w
    scale = (hi - lo) / 255.0
    zp = round(-lo / scale) - 128
    q = np.clip(np.round(w / scale) - zp, -128, 127)
    return (q + zp) * scale

model = tf.keras.models.load_model("model_tomato_mobilenetv3small_fp32.keras")
backbone = model.get_layer("MobileNetV3Small")

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

weighted_layers = [l for l in backbone.layers if len(l.get_weights()) > 0]
se_layers = [l for l in weighted_layers if "squeeze_excite" in l.name]
non_se_layers = [l for l in weighted_layers if "squeeze_excite" not in l.name]
print(f"{len(weighted_layers)} weighted layers total: {len(se_layers)} SE, {len(non_se_layers)} non-SE")

def quantize_layers(layers):
    saved = {}
    for l in layers:
        w = l.get_weights()
        saved[l.name] = [x.copy() for x in w]
        newW = [fake_quantize(x) if x.ndim > 1 else x for x in w]  # leave 1-D biases FP32
        l.set_weights(newW)
    return saved

def restore_layers(layers, saved):
    for l in layers:
        l.set_weights(saved[l.name])

results = {"baseline_fp32_acc": float(baseline_acc)}

# 1. All weighted layers quantized (sanity check vs. real TFLite INT8 13.1%)
saved = quantize_layers(weighted_layers)
acc_all = accuracy()
restore_layers(weighted_layers, saved)
print(f"All-layers weight-fake-quant accuracy: {acc_all:.4f}  (real TFLite INT8 was 0.131)")
results["all_layers_quantized_acc"] = float(acc_all)

# 2. Only SE layers quantized (rest FP32)
saved = quantize_layers(se_layers)
acc_se = accuracy()
restore_layers(se_layers, saved)
print(f"SE-layers-only quantized accuracy: {acc_se:.4f}  (drop: {baseline_acc - acc_se:.4f})")
results["se_only_quantized_acc"] = float(acc_se)

# 3. Only non-SE layers quantized (rest FP32)
saved = quantize_layers(non_se_layers)
acc_nonse = accuracy()
restore_layers(non_se_layers, saved)
print(f"Non-SE-layers-only quantized accuracy: {acc_nonse:.4f}  (drop: {baseline_acc - acc_nonse:.4f})")
results["non_se_only_quantized_acc"] = float(acc_nonse)

# 4. Per-SE-block breakdown (each block = squeeze_excite_conv + squeeze_excite_conv_1)
block_names = sorted(set(l.name.split("_squeeze_excite")[0] for l in se_layers))
block_results = {}
for bn in block_names:
    block_layers = [l for l in se_layers if l.name.startswith(bn + "_squeeze_excite")]
    saved = quantize_layers(block_layers)
    acc_b = accuracy()
    restore_layers(block_layers, saved)
    drop = baseline_acc - acc_b
    block_results[bn] = {"acc": float(acc_b), "drop": float(drop)}
    print(f"  block {bn}: acc={acc_b:.4f}  drop={drop:.4f}")
results["per_se_block"] = block_results

with open("layer_sensitivity_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved layer_sensitivity_results.json")

# 5. Depth breakdown of the non-SE layers (which the block above already
# shows carries nearly all the sensitivity): split into thirds by position
# to see whether early, middle, or late layers matter most.
n = len(non_se_layers)
thirds = {
    "early_third": non_se_layers[: n // 3],
    "middle_third": non_se_layers[n // 3 : 2 * n // 3],
    "late_third": non_se_layers[2 * n // 3 :],
}
depth_results = {}
for label, layers in thirds.items():
    saved = quantize_layers(layers)
    acc_d = accuracy()
    restore_layers(layers, saved)
    drop = baseline_acc - acc_d
    depth_results[label] = {"acc": float(acc_d), "drop": float(drop), "n_layers": len(layers)}
    print(f"  non-SE {label} ({len(layers)} layers): acc={acc_d:.4f}  drop={drop:.4f}")

results["non_se_depth_breakdown"] = depth_results
with open("layer_sensitivity_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nUpdated layer_sensitivity_results.json with depth breakdown")
