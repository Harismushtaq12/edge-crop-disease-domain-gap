"""
INT8 export of the ImageNet-pretrained (alpha=1.0) tomato model, so that the
memory audit in 52_peak_sram_tflite.py can measure it on the same converted
graph as every other backbone rather than inferring it from Keras shapes.
Identical recipe to 04_quantize_and_benchmark.py: full integer post-training
quantization, 200 randomly sampled training images for calibration.
"""
import csv, json, random
import numpy as np
import tensorflow as tf

IMG = 96
rows = [r for r in csv.DictReader(open("manifest.csv"))
        if r["is_tomato"] == "True" and r["domain"] == "pv_train"]
random.seed(42)
calib = random.sample(rows, min(200, len(rows)))

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG, IMG])
    return (img.numpy().astype(np.float32) / 127.5 - 1.0)

def rep():
    for r in calib:
        yield [np.expand_dims(load_img(r["path"]), 0)]

m = tf.keras.models.load_model("model_tomato_pretrained_fp32.keras")
c = tf.lite.TFLiteConverter.from_keras_model(m)
c.optimizations = [tf.lite.Optimize.DEFAULT]
c.representative_dataset = rep
c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
c.inference_input_type = tf.int8
c.inference_output_type = tf.int8
c.target_spec.supported_types = []
out = c.convert()
open("model_tomato_pretrained_int8.tflite", "wb").write(out)
print("wrote model_tomato_pretrained_int8.tflite", len(out), "bytes")
