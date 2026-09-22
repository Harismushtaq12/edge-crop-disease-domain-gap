"""
Extract frozen MobileNetV2 (ImageNet-pretrained) bottleneck features for every
image in manifest.csv. This is the standard transfer-learning approach used
throughout the cited literature (Howard et al., 2017; Sandler et al., 2018) and
lets us train the classification head quickly on CPU by caching the expensive
forward pass through the frozen backbone once.
"""
import csv, time, sys
import numpy as np
import tensorflow as tf

IMG_SIZE = 160  # MobileNetV2 supports 96/128/160/192/224; 160 is a common mobile-deployment resolution
BATCH = 32

print("Loading MobileNetV2 base (ImageNet weights, frozen)...")
base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet", pooling="avg"
)
base.trainable = False
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

def load_batch(paths):
    imgs = []
    for p in paths:
        img = tf.keras.utils.load_img(p, target_size=(IMG_SIZE, IMG_SIZE))
        arr = tf.keras.utils.img_to_array(img)
        imgs.append(arr)
    batch = np.stack(imgs, axis=0)
    return preprocess(batch)

rows = list(csv.DictReader(open("manifest.csv")))
print(f"Total images to process: {len(rows)}")

paths = [r["path"] for r in rows]
labels = np.array([int(r["class_idx"]) for r in rows], dtype=np.int32)
domains = np.array([r["domain"] for r in rows])

feats = np.zeros((len(rows), base.output_shape[-1]), dtype=np.float32)

t0 = time.time()
for i in range(0, len(paths), BATCH):
    batch_paths = paths[i:i + BATCH]
    x = load_batch(batch_paths)
    f = base.predict(x, verbose=0)
    feats[i:i + len(batch_paths)] = f
    if (i // BATCH) % 20 == 0:
        elapsed = time.time() - t0
        done = i + len(batch_paths)
        rate = done / max(elapsed, 1e-6)
        eta = (len(paths) - done) / max(rate, 1e-6)
        print(f"  {done}/{len(paths)}  elapsed={elapsed:.0f}s  rate={rate:.1f} img/s  ETA={eta:.0f}s", flush=True)

print(f"Done in {time.time()-t0:.0f}s")
np.savez_compressed("features.npz", feats=feats, labels=labels, domains=domains)
print("Saved features.npz  shape:", feats.shape)
