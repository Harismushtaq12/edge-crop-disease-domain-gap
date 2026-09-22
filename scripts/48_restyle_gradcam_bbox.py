"""
Restyle of the quantitative Grad-CAM leaf-box energy figure into the shared
figstyle system, so it matches the rest of the paper's figures instead of
carrying matplotlib defaults.

Two panels:
  (a) attention landing inside the annotated leaf box vs. the chance level
      (box area / image area) for all / correctly classified / misclassified
      field images, with bootstrap 95% CIs on the excess.
  (b) the per-image distribution (CDF) of energy-in-box for correct vs.
      wrong predictions, with the mean chance level marked.

Reads 18_gradcam_bbox_energy.py's saved results; recomputes nothing.
"""
import json
import numpy as np
import figstyle as fs

fs.apply()

# Scored on D_ft-test, the held-out field split, so the attention numbers sit
# on the same basis as the accuracy numbers (Section 3.7) and no arm is ever
# scored on images it adapted on. 54_gradcam_on_fttest.py does the restriction.
import csv, random
rows = [r for r in csv.DictReader(open("manifest.csv"))
        if r["is_tomato"] == "True" and r["domain"] == "pd_field"]
by = {}
for r in rows:
    by.setdefault(r["class_name"], []).append(r)
random.seed(123)
held_out = set()
for k in sorted(by):
    rs = by[k][:]
    random.shuffle(rs)
    held_out |= {r["path"].split("/")[-1] for r in rs[20:]}

d = json.load(open("gradcam_fttest_results.json"))["variants"]["baseline"]
per_image = [r for r in json.load(open("gradcam_bbox_energy_results.json"))["per_image"]
             if r["file"] in held_out]
assert len(per_image) == d["all"]["n"], (len(per_image), d["all"]["n"])

fig, axes = fs.figure(width=fs.FULL_W, height=2.6, ncols=2)

# ---- panel (a): measured attention against the chance level ---------------
groups = [("all", "All\n(n=%d)" % d["all"]["n"]),
          ("correct", "Correct\n(n=%d)" % d["correct"]["n"]),
          ("misclassified", "Wrong\n(n=%d)" % d["misclassified"]["n"])]
x = np.arange(len(groups))
w = 0.36
energy = [d[k]["mean_energy_in_box"] for k, _ in groups]
chance = [d[k]["mean_chance"] for k, _ in groups]
lo = [d[k]["excess_ci95"][0] for k, _ in groups]
hi = [d[k]["excess_ci95"][1] for k, _ in groups]
err_lo = [max(0, e - (c + l)) for e, c, l in zip(energy, chance, lo)]
err_hi = [max(0, (c + h) - e) for e, c, h in zip(energy, chance, hi)]

ax = axes[0]
b1 = ax.bar(x - w / 2, energy, width=w, color=fs.C_ATTN,
            label="Attention inside leaf box",
            yerr=[err_lo, err_hi], capsize=2.5,
            error_kw={"linewidth": 0.9, "ecolor": fs.INK})
b2 = ax.bar(x + w / 2, chance, width=w, color=fs.C_CHANCE,
            label="Chance (box area / image area)")
ax.set_xticks(x)
ax.set_xticklabels([g[1] for g in groups])
ax.set_ylabel("Fraction of Grad-CAM energy")
ax.set_ylim(0, 0.70)
fs.grid_y_only(ax)
fs.tick_labels(ax)
fs.bar_labels(ax, b1, fmt="{:.3f}", dy=0.045, fontsize=fs.FS_ANNOT)
fs.bar_labels(ax, b2, fmt="{:.3f}", dy=0.012, fontsize=fs.FS_ANNOT,
              color=fs.INK_2)
ax.legend(loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.02))
fs.panel_title(ax, "a", "Attention vs. chance, whole field test set")

# ---- panel (b): the per-image distribution behind those means -------------
ax2 = axes[1]
for key, col, label in [("correct", fs.AQUA, "Correct"),
                        ("misclassified", fs.RED, "Wrong")]:
    vals = np.sort([p["energy_in_box"] for p in per_image
                    if (p["correct"] if key == "correct" else not p["correct"])])
    if len(vals) == 0:
        continue
    y = np.arange(1, len(vals) + 1) / len(vals)
    ax2.plot(vals, y, color=col, linewidth=1.5, label=f"{label} (n={len(vals)})")

mc = d["all"]["mean_chance"]
ax2.axvline(mc, color=fs.MUTED, linestyle=(0, (3, 2)), linewidth=1.1,
            label=f"Mean chance ({mc:.2f})")
ax2.set_xlabel("Grad-CAM energy inside leaf box")
ax2.set_ylabel("Cumulative fraction of images")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.grid(True)
fs.tick_labels(ax2)
ax2.legend(loc="upper left")
fs.panel_title(ax2, "b", "Per-image distribution")

fs.finish(fig, "figures/gradcam_bbox_energy_v3.png")
