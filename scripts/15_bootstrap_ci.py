"""
Bootstrap confidence intervals on the lab-vs-field accuracy gap. Addresses
the paper's own disclosed limitation ("single-run evaluation... suggestive,
not statistically confirmed") by resampling the per-image correctness
vectors already produced by the trained INT8 models -- no retraining
needed, just re-scoring every held-out image once and then resampling in
numpy.

For each experiment (tomato, multicrop):
  - Run the INT8 model over every pv_test (lab) and pd_field (field) image,
    recording 1/0 correctness per image.
  - Bootstrap resample (10,000 iterations) each set independently to get a
    95% CI on lab accuracy, field accuracy, and (via subtraction across
    matched bootstrap iterations) the domain gap Delta_domain.
"""
import csv, json
import numpy as np
import tensorflow as tf

IMG_SIZE = 96
N_BOOT = 10000
RNG_SEED = 42

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 127.5 - 1.0
    return img.numpy().astype(np.float32)

def score_int8(model_path, rows, remap):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    correctness = np.zeros(len(rows), dtype=np.int8)
    for i, r in enumerate(rows):
        img = load_img(r["path"])
        x = np.expand_dims(img, 0)
        scale, zp = inp["quantization"]
        x = (x / scale + zp).astype(np.int8)
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        pred = int(np.argmax(interpreter.get_tensor(out["index"])[0]))
        true = remap[int(r["class_idx"])]
        correctness[i] = 1 if pred == true else 0
    return correctness

def bootstrap_ci(correctness, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(correctness)
    boot_accs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_accs[b] = correctness[idx].mean()
    point = correctness.mean()
    lo, hi = np.percentile(boot_accs, [2.5, 97.5])
    return point, lo, hi, boot_accs

results = {}
for EXPERIMENT in ["tomato", "multicrop"]:
    classmap = json.load(open(f"classmap_{EXPERIMENT}.json"))
    remap = {int(k): v for k, v in classmap["remap"].items()}
    rows = list(csv.DictReader(open("manifest.csv")))
    if EXPERIMENT == "tomato":
        rows = [r for r in rows if r["is_tomato"] == "True"]
    lab_rows = [r for r in rows if r["domain"] == "pv_test"]
    field_rows = [r for r in rows if r["domain"] == "pd_field"]

    print(f"=== {EXPERIMENT}: scoring {len(lab_rows)} lab + {len(field_rows)} field images (INT8) ===")
    lab_correct = score_int8(f"model_{EXPERIMENT}_int8.tflite", lab_rows, remap)
    field_correct = score_int8(f"model_{EXPERIMENT}_int8.tflite", field_rows, remap)

    lab_pt, lab_lo, lab_hi, lab_boot = bootstrap_ci(lab_correct)
    field_pt, field_lo, field_hi, field_boot = bootstrap_ci(field_correct, seed=RNG_SEED + 1)

    # Domain-gap CI: subtract matched-index bootstrap draws (valid for two
    # independent samples -- this is the standard two-sample bootstrap).
    gap_boot = lab_boot - field_boot
    gap_pt = lab_pt - field_pt
    gap_lo, gap_hi = np.percentile(gap_boot, [2.5, 97.5])

    print(f"  Lab   acc: {lab_pt:.4f}  95% CI [{lab_lo:.4f}, {lab_hi:.4f}]  (n={len(lab_rows)})")
    print(f"  Field acc: {field_pt:.4f}  95% CI [{field_lo:.4f}, {field_hi:.4f}]  (n={len(field_rows)})")
    print(f"  Gap (lab-field): {gap_pt:.4f}  95% CI [{gap_lo:.4f}, {gap_hi:.4f}]")
    ci_excludes_zero = gap_lo > 0
    print(f"  95% CI excludes zero (gap is statistically real): {ci_excludes_zero}")

    results[EXPERIMENT] = {
        "n_lab": len(lab_rows), "n_field": len(field_rows),
        "lab_acc": lab_pt, "lab_ci": [lab_lo, lab_hi],
        "field_acc": field_pt, "field_ci": [field_lo, field_hi],
        "gap": gap_pt, "gap_ci": [gap_lo, gap_hi],
        "gap_ci_excludes_zero": bool(ci_excludes_zero),
    }

with open("bootstrap_ci_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved bootstrap_ci_results.json")
