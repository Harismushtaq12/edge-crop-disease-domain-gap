"""
Dedicated conceptual diagram for the FARM framework (paper Section III),
requested by an external review as separate from Fig. 1's experimental
pipeline: a four-stage cyclic diagram (Feasibility -> Architecture ->
Rollout -> Maintenance) showing each stage's gating criterion and the
failure feedback loop back to Feasibility. Compact single-column figure --
this is new content added under a hard 8-page budget, so it is sized to
cost as little vertical space as possible while staying legible.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(6.6, 4.35))
ax.set_xlim(0, 100)
ax.set_ylim(0, 66)
ax.axis("off")

INK = "#242C38"
COLS = {
    "F": ("#2F6F52", "#FFFFFF"),
    "A": ("#1E3A5F", "#FFFFFF"),
    "R": ("#B4622B", "#FFFFFF"),
    "M": ("#6B4E71", "#FFFFFF"),
}
GATE_COL = "#5B6472"

def box(cx, cy, w, h, letter, title, gate, fc, tc, fontsize=9.4):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                        boxstyle="round,pad=0.34,rounding_size=1.6",
                        linewidth=1.4, edgecolor=INK, facecolor=fc, zorder=3)
    ax.add_patch(b)
    ax.text(cx, cy + h * 0.16, f"{letter} — {title}", ha="center", va="center",
            fontsize=fontsize, color=tc, weight="bold", zorder=4)
    ax.text(cx, cy - h * 0.28, gate, ha="center", va="center", fontsize=7.5,
            color=tc, zorder=4, linespacing=1.35)
    return (cx, cy, w, h)

def curved_arrow(p1, p2, rad, color=INK, lw=1.7, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15, linewidth=lw,
                                  color=color, connectionstyle=f"arc3,rad={rad}", zorder=2, linestyle=ls))

# Four stages arranged in a clockwise cycle: F top-left, A top-right,
# R bottom-right, M bottom-left.
W, H = 42, 21
F = box(25, 50, W, H, "F", "Feasibility", "Hard requirement: an independently\ncollected field-validation set (§IV-A)",
        *COLS["F"])
A = box(75, 50, W, H, "A", "Architecture", "Model family + compression\nvs. stage F's hardware budget",
        *COLS["A"])
R = box(75, 16, W, H, "R", "Rollout", "Staged, human-verified deploy;\npilot length scales with the\nlab/field gap stage F measured",
        *COLS["R"])
M = box(25, 16, W, H, "M", "Maintenance", "Continuous field accuracy +\nper-class tracking; recalibration\non season / region / strain drift",
        *COLS["M"])

def edge(a, b, rad=0.0):
    (ax1, ay1, aw, ah), (ax2, ay2, bw, bh) = a, b
    dx, dy = ax2 - ax1, ay2 - ay1
    import math
    dist = math.hypot(dx, dy)
    ux, uy = dx / dist, dy / dist
    p1 = (ax1 + ux * aw / 2 * 0.98, ay1 + uy * ah / 2 * 0.98)
    p2 = (ax2 - ux * bw / 2 * 0.98, ay2 - uy * bh / 2 * 0.98)
    curved_arrow(p1, p2, rad)

edge(F, A, rad=0.05)
edge(A, R, rad=0.05)
edge(R, M, rad=0.05)
# M -> F closing the cycle, drawn as a longer arc under both bottom boxes
curved_arrow((25 - W / 2 * 0.3, 16 - H / 2 * 0.95), (25 - W / 2 * 0.95, 50 - H / 2 * 0.3),
             rad=0.35, lw=1.7)

# Failure feedback loop: Maintenance drift finding forces a return to
# Feasibility (re-audit), drawn as a dashed loop distinct from the solid
# forward cycle.
curved_arrow((25 + W * 0.05, 16 + H / 2), (25 + W * 0.05, 50 - H / 2),
             rad=-0.55, color="#B3261E", lw=1.5, ls="--")
ax.text(51, 33, "drift / failure\nfeedback", ha="center", va="center", fontsize=7.3,
        color="#B3261E", style="italic", zorder=4)

fig.tight_layout(pad=0.3)
fig.savefig("figures/farm_diagram.png", dpi=280, bbox_inches="tight")
from PIL import Image
im = Image.open("figures/farm_diagram.png")
print("saved", im.size, "aspect", im.size[0] / im.size[1])
