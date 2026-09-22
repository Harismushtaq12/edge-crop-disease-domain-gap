"""
Resolution ablation figure: does 96x96 cause the background-attention
finding, or merely coexist with it?

The hypothesis under test is that downsampling to 96x96 destroys
lesion-scale evidence -- a 10px bacterial spot becomes ~4px, and sub-pixel in
the 6x6 feature map Grad-CAM reads -- leaving the network no option but to
key on coarse context. If so, restoring resolution should pull attention back
onto the leaf.

Panel (a) shows resolution does buy real field accuracy. Panel (b) shows it
does not buy leaf-directed attention: at 224px the model's attention is
significantly BELOW the chance level set by the leaf's own box, on correctly
classified images as well as misclassified ones. Both models start from a
resolution-matched ImageNet checkpoint and read the same named layer, so the
comparison isolates input resolution.
"""
import json
import numpy as np
import figstyle as fs
import matplotlib.pyplot as plt

fs.apply()

sb = {v["label"]: v for v in json.load(open("unified_scoreboard.json"))["variants"]}
r224 = json.load(open("results_tomato_pretrained224_fp32.json"))
g96 = json.load(open("gradcam_bbox_energy_pretrained_results.json"))
g224 = json.load(open("gradcam_bbox_energy_224_results.json"))

lab96 = sb["ImageNet-pretrained"]["lab"] * 100
fld96 = sb["ImageNet-pretrained"]["field_full"] * 100
lab224 = r224["pv_test"]["accuracy"] * 100
fld224 = r224["pd_field"]["accuracy"] * 100

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(fs.FULL_W, 2.6),
                              gridspec_kw={"width_ratios": [1.0, 1.25]})

# ------------------------------------------------- panel (a): accuracy & gap
x = np.arange(2)
w = 0.36
b1 = ax.bar(x - w / 2, [lab96, fld96], w * 0.92, color=fs.C_BEFORE,
            label="96×96", zorder=3)
b2 = ax.bar(x + w / 2, [lab224, fld224], w * 0.92, color=fs.BLUE,
            label="224×224", zorder=3)
fs.bar_labels(ax, b1, fmt="{:.1f}", fontsize=fs.FS_ANNOT - 0.6)
fs.bar_labels(ax, b2, fmt="{:.1f}", fontsize=fs.FS_ANNOT - 0.6)
ax.axhline(12.5, color=fs.MUTED, lw=0.9, ls=(0, (4, 2.5)), zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(["Lab\n(PlantVillage)", "Field\n(PlantDoc, n=744)"])
ax.set_ylabel("Top-1 accuracy (%)")
ax.set_ylim(0, 118)
fs.panel_title(ax, "a", "Resolution buys field accuracy")
ax.legend(loc="upper right", fontsize=fs.FS_LEGEND, ncol=2)
fs.grid_y_only(ax)
fs.tick_labels(ax)
ax.annotate(f"+{fld224-fld96:.1f} pp", xy=(1 + w / 2, fld224), xytext=(0, 13),
            textcoords="offset points", ha="center", fontsize=fs.FS_ANNOT - 0.4,
            color=fs.BLUE, fontweight="bold")

# --------------------------------- panel (b): attention excess, both models
groups = ["All\nimages", "Correctly\nclassified", "Mis-\nclassified"]
keys = ["all", "correct", "misclassified"]
xg = np.arange(len(groups))
for i, (d, name, col) in enumerate([(g96, "96×96", fs.C_BEFORE),
                                    (g224, "224×224", fs.BLUE)]):
    off = (i - 0.5) * w
    vals = [d[k]["mean_excess"] for k in keys]
    lo = [d[k]["mean_excess"] - d[k]["excess_ci95"][0] for k in keys]
    hi = [d[k]["excess_ci95"][1] - d[k]["mean_excess"] for k in keys]
    ax2.bar(xg + off, vals, w * 0.92, color=col, label=name, zorder=3)
    ax2.errorbar(xg + off, vals, yerr=[lo, hi], fmt="none", ecolor=fs.INK_2,
                 elinewidth=0.9, capsize=2.2, zorder=4)

ax2.axhline(0, color=fs.INK_2, lw=0.9, zorder=2)
ax2.set_xticks(xg)
ax2.set_xticklabels(groups, fontsize=fs.FS_TICK - 0.3)
ax2.set_ylabel("Attention excess over chance,\n$E(x)-C(x)$")
fs.panel_title(ax2, "b", "…but not leaf-directed attention")
ax2.set_ylim(-0.045, 0.072)
ax2.legend(loc="upper right", fontsize=fs.FS_LEGEND, ncol=2)
fs.grid_y_only(ax2)
fs.tick_labels(ax2)

fs.finish(fig, "figures/resolution_ablation.png")
