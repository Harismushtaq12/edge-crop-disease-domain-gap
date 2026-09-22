"""
Figure for the quantitative Grad-CAM bounding-box energy result
(18_gradcam_bbox_energy.py): turns the whole-field-test-set numbers into
one compact two-panel figure.
  (a) mean Grad-CAM energy landing inside the annotated leaf box vs. the
      chance level (box area / image area) for all / correct / misclassified
      field images, with bootstrap 95% CIs.
  (b) the full distribution (CDF) of per-image energy-in-box across all
      623 scored field images, with the mean chance level marked.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("gradcam_bbox_energy_results.json"))
per_image = d["per_image"]

INK = "#28313E"
FIELD_COL = "#B4622B"
CHANCE_COL = "#8A93A0"
CORRECT_COL = "#2F6F52"
WRONG_COL = "#B3261E"

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.55))

# --- panel (a): grouped bar, energy-in-box vs chance, with 95% CI --------
groups = [("all", "All\n(n=%d)" % d["all"]["n"]),
          ("correct", "Correct\n(n=%d)" % d["correct"]["n"]),
          ("misclassified", "Wrong\n(n=%d)" % d["misclassified"]["n"])]
x = np.arange(len(groups))
w = 0.34
energies = [d[k]["mean_energy_in_box"] for k, _ in groups]
chances = [d[k]["mean_chance"] for k, _ in groups]
excess_lo = [d[k]["excess_ci95"][0] for k, _ in groups]
excess_hi = [d[k]["excess_ci95"][1] for k, _ in groups]
err_lo = [max(0, e - (c + lo)) for e, c, lo in zip(energies, chances, excess_lo)]
err_hi = [max(0, (c + hi) - e) for e, c, hi in zip(energies, chances, excess_hi)]

ax = axes[0]
ax.bar(x - w / 2, energies, width=w, color=FIELD_COL, label="Grad-CAM energy\nin leaf box",
       yerr=[err_lo, err_hi], capsize=2.5, error_kw={"linewidth": 0.9, "ecolor": INK})
ax.bar(x + w / 2, chances, width=w, color=CHANCE_COL, label="Chance\n(box area / image)")
ax.set_xticks(x)
ax.set_xticklabels([g[1] for g in groups], fontsize=7.2)
ax.set_ylabel("Fraction of attention\nenergy", fontsize=7.6)
ax.tick_params(axis="y", labelsize=7)
ax.set_ylim(0, max(max(energies), max(chances)) * 1.35)
ax.legend(fontsize=6.0, loc="upper right", frameon=False, handlelength=1.2)
ax.set_title("(a) Attention vs. chance, whole field test set", fontsize=7.8)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# --- panel (b): CDF of energy_in_box, correct vs misclassified -----------
ax2 = axes[1]
for key, col, label in [("correct", CORRECT_COL, "Correct"), ("misclassified", WRONG_COL, "Wrong")]:
    vals = np.sort([p["energy_in_box"] for p in per_image if
                    (p["correct"] if key == "correct" else not p["correct"])])
    if len(vals) == 0:
        continue
    y = np.arange(1, len(vals) + 1) / len(vals)
    ax2.plot(vals, y, color=col, linewidth=1.4, label=f"{label} (n={len(vals)})")
mean_chance_all = d["all"]["mean_chance"]
ax2.axvline(mean_chance_all, color=CHANCE_COL, linestyle="--", linewidth=1.1,
            label=f"mean chance ({mean_chance_all:.2f})")
ax2.set_xlabel("Grad-CAM energy inside leaf box", fontsize=7.6)
ax2.set_ylabel("Cumulative fraction\nof images", fontsize=7.6)
ax2.tick_params(axis="both", labelsize=7)
ax2.legend(fontsize=6.0, loc="lower right", frameon=False, handlelength=1.2)
ax2.set_title("(b) Per-image distribution", fontsize=7.8)
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)

fig.tight_layout(pad=0.6, w_pad=1.4)
fig.savefig("figures/gradcam_bbox_energy.png", dpi=300, bbox_inches="tight")
print("Saved figures/gradcam_bbox_energy.png")
