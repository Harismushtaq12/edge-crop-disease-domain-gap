import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

archs = ["mobilenetv2", "mobilenetv3small", "efficientnetb0"]
labels = ["MobileNetV2", "MobileNetV3-Small", "EfficientNetB0"]

fp32 = {a: json.load(open(f"results_tomato_{a}_fp32.json")) for a in archs}
int8 = {a: json.load(open(f"results_tomato_{a}_quantized.json")) for a in archs}

fp32_lab = [fp32[a]["pv_test"]["accuracy"]*100 for a in archs]
fp32_field = [fp32[a]["pd_field"]["accuracy"]*100 for a in archs]
int8_lab = [int8[a]["int8_lab_accuracy"]*100 for a in archs]
int8_field = [int8[a]["int8_field_accuracy"]*100 for a in archs]

fig, ax = plt.subplots(figsize=(6.6, 3.4))
x = np.arange(len(labels))
w = 0.2
b1 = ax.bar(x - 1.5*w, fp32_lab, w, label="FP32 lab", color="#1f4e8c")
b2 = ax.bar(x - 0.5*w, fp32_field, w, label="FP32 field", color="#7fa8d9")
b3 = ax.bar(x + 0.5*w, int8_lab, w, label="INT8 lab", color="#c0392b")
b4 = ax.bar(x + 1.5*w, int8_field, w, label="INT8 field", color="#e8a89c")
ax.axhline(12.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.text(2.45, 14.0, "random\nguess (12.5%)", fontsize=6.5, color="gray", ha="left")

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Accuracy (%)", fontsize=9)
ax.set_title("Multi-architecture comparison: lab vs. field, FP32 vs. INT8 (tomato, shared recipe)", fontsize=9)
ax.legend(fontsize=7, ncol=2, loc="upper right")
ax.grid(alpha=0.3, axis="y")
ax.set_ylim(0, 85)

for bars, vals in [(b1, fp32_lab), (b2, fp32_field), (b3, int8_lab), (b4, int8_field)]:
    for rect, v in zip(bars, vals):
        ax.text(rect.get_x() + rect.get_width()/2, v + 1.2, f"{v:.0f}", ha="center", fontsize=6)

fig.tight_layout()
fig.savefig("figures/multiarch_comparison.png", dpi=220)
from PIL import Image
print("saved", Image.open("figures/multiarch_comparison.png").size)
