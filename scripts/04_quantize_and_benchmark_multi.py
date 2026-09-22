"""
Generalized version of 04_quantize_and_benchmark.py keyed by (experiment, arch)
tag instead of just experiment, for the multi-architecture comparison.
Usage: python3 04_quantize_and_benchmark_multi.py <experiment> <arch>
"""
import csv, json, os, sys, time, random
import numpy as np
import tensorflow as tf

EXPERIMENT = sys.argv[1]
ARCH = sys.argv[2]
TAG = f"{EXPERIMENT}_{ARCH}"
IMG_SIZE = 96

model_path = f"model_{TAG}_fp32.keras"
classmap = json.load(open(f"classmap_{TAG}.json"))
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

print("Converting to INT8 TFLite...")
tflite_int8 = converter.convert()
tflite_path = f"model_{TAG}_int8.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_int8)
int8_size = os.path.getsize(tflite_path)
print(f"FP32 size: {fp32_size/1e6:.2f} MB  INT8 size: {int8_size/1e6:.2f} MB  reduction: {fp32_size/int8_size:.2f}x")

def eval_tflite(tflite_path, rows_eval):
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
        scale, zero_point = inp["quantization"]
        x = (x / scale + zero_point).astype(np.int8)
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
    return correct / total, latencies

int8_test_acc, int8_test_lat = eval_tflite(tflite_path, test_rows)
int8_field_acc, int8_field_lat = eval_tflite(tflite_path, field_rows)
print(f"INT8 lab acc: {int8_test_acc:.4f}  field acc: {int8_field_acc:.4f}")

def stats(lats):
    arr = np.array(lats)
    return {"mean_ms": float(arr.mean()), "p50_ms": float(np.median(arr))}

results = {
    "tag": TAG, "experiment": EXPERIMENT, "architecture": ARCH, "n_classes": n_classes,
    "fp32_size_mb": fp32_size / 1e6, "int8_size_mb": int8_size / 1e6,
    "size_reduction_x": fp32_size / int8_size,
    "int8_lab_accuracy": int8_test_acc, "int8_field_accuracy": int8_field_acc,
    "latency_lab": stats(int8_test_lat), "latency_field": stats(int8_field_lat),
}
with open(f"results_{TAG}_quantized.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
