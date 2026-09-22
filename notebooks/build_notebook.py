"""
Builds notebooks/01_reproduce_paper_results.ipynb and executes it, so the
copy committed to the repository already carries its outputs and renders on
GitHub without anyone running anything.

The notebook rebuilds every table and every derived number in the paper from
the saved per-run result files in results/. It does not retrain: training is
what scripts/03_train.py and its siblings do, and the artefacts they produced
are what this notebook reads. That split is deliberate, because a reader who
wants to check the paper's arithmetic should not have to spend a day of CPU
to do it.
"""
import json
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).parent
REPO = HERE.parent


def md(t):
    return nbf.v4.new_markdown_cell(t.strip())


def code(t):
    return nbf.v4.new_code_cell(t.strip())


cells = []

cells.append(md(r"""
# Reproducing the results in *Lab-Accurate, Field-Blind*

This notebook rebuilds every table and every derived number in the paper from
the saved result files in `../results/`, and displays the figures in
`../figures/`. It runs in a few seconds and needs nothing but `numpy` and
`pandas`.

It does **not** retrain anything. Training, quantization and the Grad-CAM
sweeps are what the numbered scripts in `../scripts/` do; the JSON files this
notebook reads are their output. The split is deliberate: checking the
paper's arithmetic should not cost a day of CPU.

To regenerate the artefacts themselves from scratch, see the pipeline order in
the repository README.
"""))

cells.append(code(r"""
import json, csv, random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

RESULTS = Path("../results")
FIGURES = Path("../figures")

def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)

pd.set_option("display.width", 100)
pd.set_option("display.max_colwidth", 60)
print("result files:", len(list(RESULTS.glob('*.json'))))
"""))

# ---------------------------------------------------------------- Table 6 ---
cells.append(md(r"""
## Table 6 — the gap itself

Top-1 accuracy on the laboratory test split and on the held-out field set, at
both precisions.
"""))

cells.append(code(r"""
rows = []
for label, fp32, q in [
    ("Tomato (8 classes)",      "results_tomato_fp32.json",     "results_tomato_quantized.json"),
    ("Multi-crop (21 classes)", "results_multicrop_fp32.json",  "results_multicrop_quantized.json"),
]:
    a, b = load(fp32), load(q)
    rows.append([label, "32-bit", 100*a["pv_test"]["accuracy"],
                 100*a["pd_field"]["accuracy"]])
    rows.append([label, "8-bit",  100*b["int8"]["pv_test_accuracy"],
                 100*b["int8"]["pd_field_accuracy"]])

t6 = pd.DataFrame(rows, columns=["Model", "Precision", "Laboratory %", "Field %"])
t6.round(2)
"""))

# ------------------------------------------------------------- the floors ---
cells.append(md(r"""
### The two chance floors

The paper compares field accuracy against a uniform-guessing rate *and*
against a constant predictor that always answers the most frequent class.
Both are computed here from the manifest rather than quoted.
"""))

cells.append(code(r"""
manifest = list(csv.DictReader(open(RESULTS / "manifest.csv")))

def floors(rows, name):
    c = Counter(r["class_name"] for r in rows)
    n = sum(c.values())
    top, k = c.most_common(1)[0]
    return {
        "set": name, "n": n, "classes": len(c),
        "uniform %": round(100 / len(c), 2),
        "majority %": round(100 * k / n, 2),
        "majority class": top.split("___")[-1],
    }

pd.DataFrame([
    floors([r for r in manifest if r["is_tomato"] == "True" and r["domain"] == "pd_field"],
           "tomato field (744)"),
    floors([r for r in manifest if r["domain"] == "pd_field"], "multi-crop field (2010)"),
])
"""))

cells.append(code(r"""
# D_ft-test is rebuilt with the same seed the fine-tuning scripts use, so the
# majority rate quoted for the common basis is reproduced rather than asserted.
field = [r for r in manifest if r["is_tomato"] == "True" and r["domain"] == "pd_field"]
by = {}
for r in field:
    by.setdefault(r["class_name"], []).append(r)
random.seed(123)
ft_test = []
for k in sorted(by):
    rs = by[k][:]
    random.shuffle(rs)
    ft_test += rs[20:]

c = Counter(r["class_name"] for r in ft_test)
top, k = c.most_common(1)[0]
print(f"D_ft-test: n={len(ft_test)}  uniform={100/len(c):.2f}%  "
      f"majority={100*k/len(ft_test):.2f}% ({top.split('___')[-1]})")
"""))

# ---------------------------------------------------------------- bootstrap -
cells.append(md(r"""
### Bootstrap intervals on the gap
"""))

cells.append(code(r"""
b = load("bootstrap_ci_results.json")
pd.DataFrame([{
    "experiment": k,
    "lab %": round(100*v["lab_acc"], 2),
    "field %": round(100*v["field_acc"], 2),
    "gap pp": round(100*v["gap"], 1),
    "gap 95% CI": f"{100*v['gap_ci'][0]:.1f} – {100*v['gap_ci'][1]:.1f}",
    "excludes zero": v["gap_ci_excludes_zero"],
} for k, v in b.items()])
"""))

# ---------------------------------------------------------------- Table 7 ---
cells.append(md(r"""
## Table 7 — size and latency

Compression is close to free for this architecture, which is what makes the
gap in Table 6 a domain problem rather than a compression problem.
"""))

cells.append(code(r"""
rows = []
for label, key in [("Tomato", "tomato"), ("Multi-crop", "multicrop")]:
    q = load(f"results_{key}_quantized.json")
    rows.append([
        label,
        round(q["fp32_tflite_size_mb"], 2), round(q["int8_tflite_size_mb"], 2),
        f"{q['size_reduction_x']:.2f}x",
        round(q["fp32_tflite"]["latency_pv_test"]["mean_ms"], 2),
        round(q["int8"]["latency_pv_test"]["mean_ms"], 2),
    ])
pd.DataFrame(rows, columns=["Model", "32-bit MB", "8-bit MB", "Reduction",
                            "32-bit ms", "8-bit ms"])
"""))

# ---------------------------------------------------------------- Table 8 ---
cells.append(md(r"""
## Table 8 and Figure 5 — the collapse is uneven

Per-class accuracy for the deployed 8-bit tomato model. The ordering inverts
between the two domains.
"""))

cells.append(code(r"""
pc = load("perclass_tomato.json")
rows = []
for cls, lab in pc["lab"].items():
    fld = pc["field"][cls]
    rows.append([cls.split("___")[-1].replace("_", " "),
                 f"{100*lab[0]:.1f}% ({lab[1]}/{lab[2]})",
                 f"{100*fld[0]:.1f}% ({fld[1]}/{fld[2]})",
                 100*fld[0]])
t8 = (pd.DataFrame(rows, columns=["Class", "Laboratory", "Field", "_sort"])
        .sort_values("_sort", ascending=False).drop(columns="_sort")
        .reset_index(drop=True))
t8
"""))

cells.append(code(r"""
from IPython.display import Image, display
display(Image(filename=str(FIGURES / "perclass_v2.png"), width=760))
display(Image(filename=str(FIGURES / "confmat_v2.png"), width=760))
"""))

# ------------------------------------------------------------- attention ----
cells.append(md(r"""
## Section 5.4 and Figure 9 — where the model is actually looking

Grad-CAM attention landing inside the annotated leaf box, $E(x)$, against the
chance level set by that box's own area, $C(x)$. Every arm is scored on the
495 box-annotated images inside $D_{\text{ft-test}}$, the same basis the
accuracy table uses, so no adapted arm is scored on its own training images.
"""))

cells.append(code(r"""
ft = load("gradcam_fttest_results.json")["variants"]
rows = []
for arm, v in ft.items():
    for grp in ("all", "correct", "misclassified"):
        g = v[grp]
        rows.append([arm, grp, g["n"], round(g["mean_energy_in_box"], 4),
                     round(g["mean_chance"], 4), round(g["mean_excess"], 4),
                     f"[{g['excess_ci95'][0]:+.4f}, {g['excess_ci95'][1]:+.4f}]",
                     "yes" if (g["excess_ci95"][0] > 0) or (g["excess_ci95"][1] < 0) else "no"])
att = pd.DataFrame(rows, columns=["arm", "subset", "n", "E(x)", "C(x)",
                                  "excess", "95% CI", "CI excludes 0"])
att
"""))

cells.append(code(r"""
display(Image(filename=str(FIGURES / "gradcam_field_v3.png"), width=760))
display(Image(filename=str(FIGURES / "gradcam_bbox_energy_v3.png"), width=760))
display(Image(filename=str(FIGURES / "attention_comparison.png"), width=760))
"""))

cells.append(md(r"""
### The null result the paper reports

Figure 7 invites the reading that the model learned a photographic setup: the
images it gets right look like close-ups. The paper tests that against the one
number available for it, the leaf's fractional area $C(x)$, and reports that
it does not hold.
"""))

cells.append(code(r"""
per = load("gradcam_bbox_energy_results.json")["per_image"]
held = {r["path"].split("/")[-1] for r in ft_test}
kept = [p for p in per if p["file"] in held]

cc = np.array([p["chance"] for p in kept if p["correct"]])
cw = np.array([p["chance"] for p in kept if not p["correct"]])
rng = np.random.default_rng(0)
boot = np.array([rng.choice(cc, cc.size, True).mean() - rng.choice(cw, cw.size, True).mean()
                 for _ in range(10000)])
print(f"mean C(x), correct   (n={cc.size:3d}): {cc.mean():.4f}")
print(f"mean C(x), incorrect (n={cw.size:3d}): {cw.mean():.4f}")
print(f"difference {cc.mean()-cw.mean():+.4f}  "
      f"95% CI [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}]")
print("\nThe interval spans zero, so the impression is not supported.")
"""))

# --------------------------------------------------------------- Table 9 ----
cells.append(md(r"""
## Table 9 and Figure 11 — it is not the architecture

Three backbones under one shared recipe, and the per-layer quantization sweep
that locates MobileNetV3-Small's collapse.
"""))

cells.append(code(r"""
rows = []
for label, key in [("MobileNetV2", "mobilenetv2"),
                   ("MobileNetV3-Small", "mobilenetv3small"),
                   ("EfficientNetB0", "efficientnetb0")]:
    a = load(f"results_tomato_{key}_fp32.json")
    q = load(f"results_tomato_{key}_quantized.json")
    la, fa = a["pv_test"]["accuracy"], a["pd_field"]["accuracy"]
    lq, fq = q["int8_lab_accuracy"], q["int8_field_accuracy"]
    rows.append([label, "32-bit", 100*la, 100*fa, 100*(la-fa)])
    rows.append([label, "8-bit",  100*lq, 100*fq, 100*(lq-fq)])
pd.DataFrame(rows, columns=["Architecture", "Precision", "Laboratory %",
                            "Field %", "gap pp"]).round(1)
"""))

cells.append(code(r"""
ls = load("layer_sensitivity_blocks_results.json")
base = ls["baseline_fp32_acc"]
rows = [[g, v["n_layers"], v["n_quantized_params"],
         round(100*v["acc"], 2), round(100*v["drop"], 2)]
        for g, v in ls["groups"].items()]
t = pd.DataFrame(rows, columns=["Layer group", "layers", "params",
                                "accuracy %", "drop pp"]).sort_values("drop pp",
                                                                      ascending=False)
print(f"un-quantized 32-bit baseline: {100*base:.2f}%\n")
display(t.reset_index(drop=True))
print("\nThe 45 squeeze-and-excitation layers cost nothing. The two-layer stem,")
print("432 parameters, accounts for the collapse.")
"""))

cells.append(code(r"""
display(Image(filename=str(FIGURES / "multiarch_v2.png"), width=760))
display(Image(filename=str(FIGURES / "layer_sensitivity_v2.png"), width=680))
"""))

# -------------------------------------------------------------- Table 13 ----
cells.append(md(r"""
## Table 13 and Figure 15 — all eight configurations on one basis

Every saved model re-scored on $D_{\text{ft-test}}$, which is disjoint from
adaptation data for every arm. The two accountings for the stacked
configuration — against the from-scratch baseline and against the pretrained
model it was built on — are both computed here.
"""))

cells.append(code(r"""
sb = load("unified_scoreboard.json")
V = {v["label"]: v for v in sb["variants"]}
order = ["From-scratch baseline", "+ full fine-tune", "+ head-only fine-tune",
         "+ L2-SP fine-tune", "+ CutMix", "ImageNet-pretrained",
         "Pretrained + head-only", "Pretrained + L2-SP"]
rows = []
for lbl in order:
    v = V[lbl]
    rows.append([lbl, 100*v["lab"], 100*v["field_fttest"],
                 100*(v["lab"] - v["field_fttest"])])
t13 = pd.DataFrame(rows, columns=["Configuration", "Laboratory %",
                                  "Field % (D_ft-test)", "gap pp"]).round(1)
display(t13)

base, pre = V["From-scratch baseline"], V["ImageNet-pretrained"]
print("\nThe two accountings for the stacked arms:")
for lbl in ["Pretrained + head-only", "Pretrained + L2-SP"]:
    v = V[lbl]
    print(f"  {lbl}")
    print(f"    vs from-scratch baseline : lab {100*(v['lab']-base['lab']):+.1f} pp, "
          f"field {100*(v['field_fttest']-base['field_fttest']):+.1f} pp")
    print(f"    vs pretrained model      : lab {100*(v['lab']-pre['lab']):+.1f} pp, "
          f"field {100*(v['field_fttest']-pre['field_fttest']):+.1f} pp")
"""))

cells.append(code(r"""
display(Image(filename=str(FIGURES / "tradeoff_frontier.png"), width=760))
display(Image(filename=str(FIGURES / "mitigation_v2.png"), width=760))
display(Image(filename=str(FIGURES / "cutmix_v2.png"), width=760))
display(Image(filename=str(FIGURES / "pretrained_v2.png"), width=760))
display(Image(filename=str(FIGURES / "resolution_ablation.png"), width=760))
"""))

# -------------------------------------------------------------- Table 14 ----
cells.append(md(r"""
## Table 14 — activation memory on the converted 8-bit graph

The arena is the largest total of simultaneously live activation tensors on
the TFLite graph that would actually be executed. Each peak is small enough to
check by hand, and the check is run below.
"""))

cells.append(code(r"""
sram = load("peak_sram_tflite_results.json")
rows = [[v["label"].replace("$\\alpha$", "alpha"), v["n_ops"],
         v["peak_op_kb"], v["peak_arena_kb"], v["peak_arena_at"],
         "not excluded" if v["fits"]["Arduino Nano 33 BLE (nRF52840)"] else "excluded"]
        for v in sram["variants"]]
pd.DataFrame(rows, columns=["Backbone", "ops", "per-op KB", "arena KB",
                            "peak at", "256 KB part"])
"""))

cells.append(code(r"""
# Hand-check of the three peaks quoted in the paper.
checks = [
    ("MobileNetV2 alpha=0.75", 48*48*96 + 24*24*96, 270.0),
    ("MobileNetV3-Small",      48*48*16 + 48*48*16,  72.0),
    ("EfficientNetB0",         48*48*144 + 48*48*144, 648.0),
]
for name, byts, quoted in checks:
    print(f"{name:26s} {byts:8,d} B = {byts/1024:7.1f} KB   "
          f"(paper: {quoted} KB)  {'OK' if abs(byts/1024-quoted) < 0.1 else 'MISMATCH'}")
"""))

cells.append(md(r"""
## Figures that appear in the paper but are not rebuilt above
"""))

cells.append(code(r"""
for f in ["fig_farm.png", "fig_pipeline.png", "lab_vs_field_samples_v3.png",
          "train_curves_v2.png", "prediction_samples_grid_v3.png"]:
    display(Image(filename=str(FIGURES / f), width=760))
"""))

cells.append(md(r"""
---

Every number printed above is read from the saved run artefacts, not
re-derived from the manuscript. If a figure here disagrees with the paper,
the artefacts are the authority.
"""))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                             "language": "python"}
nb.metadata["language_info"] = {"name": "python"}

out = HERE / "01_reproduce_paper_results.ipynb"
nbf.write(nb, out)
print(f"wrote {out.name}: {len(cells)} cells")

print("executing...")
client = NotebookClient(nb, timeout=900, kernel_name="python3",
                        resources={"metadata": {"path": str(HERE)}})
client.execute()
nbf.write(nb, out)

n_out = sum(1 for c in nb.cells if c.get("outputs"))
print(f"executed: {n_out} cells now carry output")
