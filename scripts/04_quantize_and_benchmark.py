"""
Convert a trained Keras model to TFLite with INT8 post-training quantization,
then:
  1. Re-measure accuracy on PlantVillage test (lab) and PlantDoc (field) using
     the actual quantized TFLite interpreter (not the original Keras model).
  2. Benchmark CPU inference latency of the quantized model (single-image,
     repeated timed calls) on this workspace's CPU, honestly labeled as a
     desktop/server CPU benchmark rather than true embedded-device latency.
  3. Report model file size before/after quantization.
"""
import csv, json, os, sys, time
import numpy as np
import tensorflow as tf

EXPERIMENT = sys.argv[1] if len(sys.argv) > 1 else "multicrop"
IMG_SIZE = 96

model_path = f"model_{EXPERIMENT}_fp32.keras"
classmap = json.load(open(f"classmap_{EXPERIMENT}.json"))
remap = {int(k): v for k, v in classmap["remap"].items()}
n_classes = len(classmap["class_ids"])

rows = list(csv.DictReader(open("manifest.csv")))
if EXPERIMENT == "tomato":
    rows = [r for r in rows if r["is_tomato"] == "True"]

def get_rows(domain):
    return [r for r in rows if r["domain"] == domain]

train_rows = get_rows("pv_train")
test_rows = get_rows("pv_test")
field_rows = get_rows("pd_field")

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 127.5 - 1.0
    return img.numpy().astype(np.float32)

print("Loading FP32 Keras model:", model_path)
keras_model = tf.keras.models.load_model(model_path)

fp32_size = os.path.getsize(model_path)

# --- representative dataset for INT8 calibration (subset of training images) ---
import random
random.seed(42)
calib_rows = random.sample(train_rows, min(200, len(train_rows)))

def representative_dataset():
    for r in calib_rows:
        img = load_img(r["path"])
        yield [np.expand_dims(img, 0)]

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.target_spec.supported_types = []

print("Converting to INT8 TFLite (this runs the calibration pass)...")
tflite_int8 = converter.convert()
tflite_path = f"model_{EXPERIMENT}_int8.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_int8)
int8_size = os.path.getsize(tflite_path)

print(f"FP32 Keras model size: {fp32_size/1e6:.2f} MB")
print(f"INT8 TFLite model size: {int8_size/1e6:.2f} MB")
print(f"Size reduction: {fp32_size/int8_size:.2f}x")

# --- also export a plain FP32 TFLite for apples-to-apples latency comparison ---
converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_fp32 = converter_fp32.convert()
fp32_tflite_path = f"model_{EXPERIMENT}_fp32.tflite"
with open(fp32_tflite_path, "wb") as f:
    f.write(tflite_fp32)
fp32_tflite_size = os.path.getsize(fp32_tflite_path)

def eval_tflite(tflite_path, rows_eval, quant):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    correct = 0
    total = 0
    latencies = []
    for r in rows_eval:
        img = load_img(r["path"])
        x = np.expand_dims(img, 0)
        if quant:
            scale, zero_point = inp["quantization"]
            x = (x / scale + zero_point).astype(np.int8)
        else:
            x = x.astype(np.float32)
        interpreter.set_tensor(inp["index"], x)
        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        pred = interpreter.get_tensor(out["index"])[0]
        pred_class = int(np.argmax(pred))
        true_class = remap[int(r["class_idx"])]
        if pred_class == true_class:
            correct += 1
        total += 1
    acc = correct / total
    return acc, latencies

print("\nEvaluating INT8 TFLite model...")
int8_test_acc, int8_test_lat = eval_tflite(tflite_path, test_rows, quant=True)
int8_field_acc, int8_field_lat = eval_tflite(tflite_path, field_rows, quant=True)
print(f"  INT8 PlantVillage test (lab) acc: {int8_test_acc:.4f}")
print(f"  INT8 PlantDoc field acc:          {int8_field_acc:.4f}")

print("\nEvaluating FP32 TFLite model (for latency comparison)...")
fp32_test_acc, fp32_test_lat = eval_tflite(fp32_tflite_path, test_rows, quant=False)
fp32_field_acc, fp32_field_lat = eval_tflite(fp32_tflite_path, field_rows, quant=False)
print(f"  FP32-TFLite PlantVillage test (lab) acc: {fp32_test_acc:.4f}")
print(f"  FP32-TFLite PlantDoc field acc:          {fp32_field_acc:.4f}")

def stats(lats):
    arr = np.array(lats)
    return {"mean_ms": float(arr.mean()), "p50_ms": float(np.median(arr)), "p95_ms": float(np.percentile(arr, 95))}

results = {
    "experiment": EXPERIMENT,
    "n_classes": n_classes,
    "img_size": IMG_SIZE,
    "fp32_keras_size_mb": fp32_size / 1e6,
    "fp32_tflite_size_mb": fp32_tflite_size / 1e6,
    "int8_tflite_size_mb": int8_size / 1e6,
    "size_reduction_x": fp32_tflite_size / int8_size,
    "int8": {
        "pv_test_accuracy": int8_test_acc,
        "pd_field_accuracy": int8_field_acc,
        "latency_pv_test": stats(int8_test_lat),
        "latency_pd_field": stats(int8_field_lat),
    },
    "fp32_tflite": {
        "pv_test_accuracy": fp32_test_acc,
        "pd_field_accuracy": fp32_field_acc,
        "latency_pv_test": stats(fp32_test_lat),
        "latency_pd_field": stats(fp32_field_lat),
    },
    "note": "Latency measured via TFLite Interpreter single-image invoke() calls on this workspace's CPU (2 vCPU, no GPU/NPU) -- a desktop/server CPU benchmark, not measured on physical embedded/microcontroller hardware.",
}

with open(f"results_{EXPERIMENT}_quantized.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + json.dumps(results, indent=2))
