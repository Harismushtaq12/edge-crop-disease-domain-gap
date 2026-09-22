import csv, json, sys
import numpy as np
import tensorflow as tf

EXPERIMENT = sys.argv[1] if len(sys.argv) > 1 else "tomato"
IMG_SIZE = 96

classmap = json.load(open(f"classmap_{EXPERIMENT}.json"))
remap = {int(k): v for k, v in classmap["remap"].items()}
inv_remap = {v: k for k, v in remap.items()}

rows = list(csv.DictReader(open("manifest.csv")))
if EXPERIMENT == "tomato":
    rows = [r for r in rows if r["is_tomato"] == "True"]

# class idx -> name
idx_to_name = {}
for r in rows:
    idx_to_name[int(r["class_idx"])] = r["class_name"]

field_rows = [r for r in rows if r["domain"] == "pd_field"]
test_rows = [r for r in rows if r["domain"] == "pv_test"]

interpreter = tf.lite.Interpreter(model_path=f"model_{EXPERIMENT}_int8.tflite")
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 127.5 - 1.0
    return img.numpy().astype(np.float32)

def per_class_acc(rows_eval):
    correct = {}
    total = {}
    for r in rows_eval:
        img = load_img(r["path"])
        x = np.expand_dims(img, 0)
        scale, zero_point = inp["quantization"]
        x = (x / scale + zero_point).astype(np.int8)
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        pred = interpreter.get_tensor(out["index"])[0]
        pred_class = int(np.argmax(pred))
        true_class = remap[int(r["class_idx"])]
        cname = idx_to_name[int(r["class_idx"])]
        total[cname] = total.get(cname, 0) + 1
        if pred_class == true_class:
            correct[cname] = correct.get(cname, 0) + 1
    return {k: (correct.get(k, 0) / v, correct.get(k, 0), v) for k, v in total.items()}

print(f"=== {EXPERIMENT}: per-class field (PlantDoc) accuracy, INT8 model ===")
field_acc = per_class_acc(field_rows)
for k, (acc, c, t) in sorted(field_acc.items(), key=lambda x: -x[1][0]):
    print(f"  {acc:.3f}  ({c:3d}/{t:3d})  {k}")

print(f"\n=== {EXPERIMENT}: per-class lab (PlantVillage test) accuracy, INT8 model ===")
test_acc = per_class_acc(test_rows)
for k, (acc, c, t) in sorted(test_acc.items(), key=lambda x: -x[1][0]):
    print(f"  {acc:.3f}  ({c:3d}/{t:3d})  {k}")

with open(f"perclass_{EXPERIMENT}.json", "w") as f:
    json.dump({"field": field_acc, "lab": test_acc}, f, indent=2)
