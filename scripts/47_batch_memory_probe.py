"""
Measures the memory cost of training MobileNetV2 at 224x224 as a function of
batch size, so the paper's claim that batch 64 was not affordable is a
measurement rather than an anecdote.

Why this exists. Section 4.11 reports that the 224x224 arm uses batch 16
rather than the 96x96 arm's batch 64, and attributes that to memory. A
reviewer is entitled to ask how much memory, and whether the reduction was
actually forced or merely convenient. This script answers both by measuring
peak resident set size (VmHWM, the kernel's own high-water mark) for a real
forward+backward pass at each batch size, and recording whether the process
survived.

Design: each batch size runs in its own child process. That matters for two
reasons -- an out-of-memory kill then terminates only the child, leaving the
sweep able to continue and record the failure as data; and each measurement
starts from a clean allocator, so peaks are not inherited from the previous
run.

The probe builds the SAME tf.data input pipeline as 33_resolution_ablation_224.py,
including its shuffle(2048) buffer, rather than feeding a single synthetic
tensor. This is not incidental: an earlier version of this probe used synthetic
tensors, reported that batch 64 peaked at 5.94 GB and survived, and was then
contradicted by the real training run, which the OOM killer terminated at
5.97 GB. The difference is the input pipeline -- a 2048-element shuffle buffer
of decoded 224x224x3 float images is itself ~1.2 GB, and a probe that omits it
under-reports the footprint by exactly the amount that decides the outcome.
The lesson generalises: measure the pipeline you actually run.

The reported peak is the whole training-process footprint (weights,
optimizer slots, activations, the TensorFlow runtime and its allocator
arena), which is the number that actually decides whether a run survives --
not the pure activation figure computed analytically in
38_peak_sram_analysis.py, which answers a different question (what a
microcontroller inference arena would need).

Usage: python3 47_batch_memory_probe.py            # runs the sweep
       python3 47_batch_memory_probe.py --child 32 # one measurement
"""
import json
import os
import subprocess
import sys

IMG_SIZE = 224
STEPS = 12   # enough for the shuffle buffer to fill
BATCHES = [8, 16, 32, 64]
OUT = "batch_memory_probe.json"


def peak_rss_mb():
    """Kernel high-water mark for this process, in MB."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def available_mb():
    with open("/proc/meminfo") as f:
        info = {l.split(":")[0]: l.split()[1] for l in f if ":" in l}
    return int(info["MemAvailable"]) / 1024.0, int(info["MemTotal"]) / 1024.0


def run_child(batch):
    """A few real training steps on the real input pipeline; print peak RSS."""
    import csv
    import numpy as np
    import tensorflow as tf
    sys.path.insert(0, ".")
    import importlib.util
    # 28_ parses sys.argv at import time (img_size, checkpoint, output), so the
    # child's own flags must be hidden from it or it reads "--child" as a size.
    saved_argv, sys.argv = sys.argv, [sys.argv[0], str(IMG_SIZE)]
    spec = importlib.util.spec_from_file_location("cvt", "28_convert_pretrained_weights.py")
    cvt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cvt)
    sys.argv = saved_argv

    from tensorflow.keras import layers, Model
    full, _ = cvt.build_keras_mobilenetv2(IMG_SIZE, num_classes=1000, with_top=True)
    feat = full.get_layer("avgpool").output
    out = layers.Dense(8, activation="softmax", name="tomato_head")(feat)
    model = Model(full.input, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                  loss="sparse_categorical_crossentropy")

    # --- the real input pipeline, shuffle buffer included ---
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rows = [r for r in csv.DictReader(open("manifest.csv"))
            if r["is_tomato"] == "True" and r["domain"] == "pv_train"]
    class_ids = sorted({int(r["class_idx"]) for r in rows})
    remap = {o: n for n, o in enumerate(class_ids)}
    paths = [r["path"] for r in rows]
    labels = [remap[int(r["class_idx"])] for r in rows]

    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE]) / 255.0
        return img, label

    ds = (tf.data.Dataset.from_tensor_slices((paths, labels))
          .map(load, num_parallel_calls=tf.data.AUTOTUNE)
          .shuffle(2048, seed=42)                       # the memory that matters
          .map(lambda i, l: ((i - MEAN) / STD, l), num_parallel_calls=tf.data.AUTOTUNE)
          .batch(batch).prefetch(tf.data.AUTOTUNE))

    for i, (x, y) in enumerate(ds):
        model.train_on_batch(x, y)          # real forward + backward + update
        if i + 1 >= STEPS:
            break

    print("PEAK_RSS_MB=%.1f" % peak_rss_mb())


if "--child" in sys.argv:
    run_child(int(sys.argv[sys.argv.index("--child") + 1]))
    raise SystemExit(0)

avail, total = available_mb()
print(f"container memory: {total:.0f} MB total, {avail:.0f} MB available at probe start")
print(f"probing {IMG_SIZE}x{IMG_SIZE} training, {STEPS} steps per batch size\n")

rows = []
for b in BATCHES:
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
    p = subprocess.run([sys.executable, __file__, "--child", str(b)],
                       capture_output=True, text=True, env=env)
    peak = None
    for line in p.stdout.splitlines():
        if line.startswith("PEAK_RSS_MB="):
            peak = float(line.split("=")[1])
    killed = p.returncode != 0
    # 137 = 128+SIGKILL, which is what the OOM killer leaves behind
    oom = p.returncode in (137, -9)
    rows.append({"batch": b, "peak_rss_mb": peak, "returncode": p.returncode,
                 "survived": not killed, "oom_killed": oom})
    status = "OK" if not killed else ("OOM-KILLED" if oom else f"failed rc={p.returncode}")
    peak_s = f"{peak:8.1f} MB" if peak else "       -- "
    print(f"  batch {b:3d}: peak RSS {peak_s}   {status}")
    if killed and not oom:
        print("    stderr tail:", p.stderr.strip().splitlines()[-1][:160] if p.stderr.strip() else "(none)")

ok = [r for r in rows if r["survived"] and r["peak_rss_mb"]]
if len(ok) >= 2:
    per_img = (ok[-1]["peak_rss_mb"] - ok[0]["peak_rss_mb"]) / (ok[-1]["batch"] - ok[0]["batch"])
    fixed = ok[0]["peak_rss_mb"] - per_img * ok[0]["batch"]
    print(f"\nfit over surviving runs: ~{fixed:.0f} MB fixed + ~{per_img:.1f} MB per image")
    print(f"extrapolated batch 64: ~{fixed + per_img*64:.0f} MB")

json.dump({"img_size": IMG_SIZE, "steps": STEPS,
           "container_total_mb": round(total), "container_available_mb": round(avail),
           "measurements": rows}, open(OUT, "w"), indent=2)
print(f"\nDONE -> {OUT}")
