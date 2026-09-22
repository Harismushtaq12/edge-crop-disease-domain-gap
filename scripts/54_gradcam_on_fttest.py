"""
Re-score the Grad-CAM leaf-box attention metric on D_ft-test only.

Why this exists. The attention metric was computed over all 623 scorable
tomato field images. For the arms that use field data, 160 of those 744
images were consumed as D_ft-train / D_ft-val, so up to 160 of the scored
images are images those arms trained on. Section 3.7 already puts accuracy
on the single common basis D_ft-test for exactly this reason; the attention
numbers must sit on the same basis or the two halves of the paper are
measuring different populations.

This is a re-analysis: it reuses the per-image energies already computed by
18/26/30/40/45_*, and simply restricts them to the held-out split and
re-runs the bootstrap. Nothing is recomputed from images.
"""
import csv, json, random
import numpy as np

# Rebuild D_ft-test exactly as 07_finetune_mitigation.py does (same seed).
rows = [r for r in csv.DictReader(open("manifest.csv"))
        if r["is_tomato"] == "True" and r["domain"] == "pd_field"]
by = {}
for r in rows:
    by.setdefault(r["class_name"], []).append(r)
random.seed(123)
ft_test = []
for k in sorted(by):
    rs = by[k][:]
    random.shuffle(rs)
    ft_test += rs[20:]
held_out = {r["path"].split("/")[-1] for r in ft_test}
print(f"D_ft-test: {len(ft_test)} images")

SOURCES = [
    ("baseline",   "gradcam_bbox_energy_results.json"),
    ("cutmix",     "gradcam_bbox_energy_cutmix_results.json"),
    ("pretrained", "gradcam_bbox_energy_pretrained_results.json"),
    ("stacked",    "gradcam_bbox_energy_stacked_results.json"),
    ("res224",     "gradcam_bbox_energy_224_results.json"),
]

rng = np.random.default_rng(20260921)


def summarise(recs):
    if not recs:
        return None
    e = np.array([r["energy_in_box"] for r in recs])
    c = np.array([r["chance"] for r in recs])
    d = e - c
    boot = np.array([rng.choice(d, d.size, True).mean() for _ in range(10000)])
    return {
        "n": int(d.size),
        "mean_energy_in_box": float(e.mean()),
        "mean_chance": float(c.mean()),
        "mean_excess": float(d.mean()),
        "excess_ci95": [float(np.percentile(boot, 2.5)),
                        float(np.percentile(boot, 97.5))],
    }


out = {"basis": "D_ft-test (584 field images) intersected with box-annotated",
       "variants": {}}
for name, path in SOURCES:
    try:
        per = json.load(open(path))["per_image"]
    except FileNotFoundError:
        print("  missing:", path)
        continue
    kept = [r for r in per if r["file"] in held_out]
    rec = {
        "n_scored_all": len(per),
        "n_scored_fttest": len(kept),
        "all": summarise(kept),
        "correct": summarise([r for r in kept if r["correct"]]),
        "misclassified": summarise([r for r in kept if not r["correct"]]),
    }
    out["variants"][name] = rec
    a, co, w = rec["all"], rec["correct"], rec["misclassified"]
    print(f"{name:11s} kept {len(kept):3d}/{len(per):3d}   "
          f"all {a['mean_excess']:+.4f} [{a['excess_ci95'][0]:+.4f},{a['excess_ci95'][1]:+.4f}]  "
          f"correct n={co['n']:3d} {co['mean_excess']:+.4f}  "
          f"wrong n={w['n']:3d} {w['mean_excess']:+.4f}")

json.dump(out, open("gradcam_fttest_results.json", "w"), indent=2)
print("DONE -> gradcam_fttest_results.json")
