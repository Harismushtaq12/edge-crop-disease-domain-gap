import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

full = json.load(open("results_tomato_finetune_mitigation.json"))
head = json.load(open("results_tomato_headft_mitigation.json"))

labels = ["Before", "Full fine-tune", "Head-only fine-tune"]
field_acc = [full["before"]["ft_test_accuracy"]*100, full["after"]["ft_test_accuracy"]*100, head["after"]["ft_test_accuracy"]*100]
lab_acc = [full["before"]["lab_accuracy"]*100, full["after"]["lab_accuracy"]*100, head["after"]["lab_accuracy"]*100]

fig, ax = plt.subplots(figsize=(4.6, 3.0))
x = np.arange(len(labels))
w = 0.35
ax.bar(x - w/2, lab_acc, w, label="Lab accuracy", color="#1f4e8c")
ax.bar(x + w/2, field_acc, w, label="Field accuracy (ft_test)", color="#c0392b")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Few-shot field fine-tuning: lab vs. field accuracy", fontsize=9)
ax.legend(fontsize=7)
ax.grid(alpha=0.3, axis="y")
for i, v in enumerate(lab_acc):
    ax.text(i - w/2, v + 1.5, f"{v:.0f}", ha="center", fontsize=7)
for i, v in enumerate(field_acc):
    ax.text(i + w/2, v + 1.5, f"{v:.0f}", ha="center", fontsize=7)
fig.tight_layout()
fig.savefig("figures/mitigation_comparison.png", dpi=220)
print("wrote figures/mitigation_comparison.png")
from PIL import Image
print(Image.open("figures/mitigation_comparison.png").size)
