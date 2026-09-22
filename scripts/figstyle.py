"""
Shared figure style for every figure in the paper.

The point of this module is that the paper's figures should read as one
designed system rather than a dozen separately-improvised plots: one
typeface stack, one type scale, one palette, one set of spacing and
axis-chrome rules, applied everywhere.

Palette provenance: the categorical hues below are a CVD-validated set
(protan/deutan/tritan separation checked pairwise against a white print
surface, worst adjacent pair dE 9.1 protan / 22.9 normal vision). They are
used in fixed slot order and never cycled. Because two of the slots sit
below 3:1 contrast on white, every chart that uses them also carries direct
value labels, so colour never carries meaning on its own -- which is also
what makes the figures survive greyscale printing and colour-blind readers.

Usage:
    import figstyle as fs
    fs.apply()
    fig, ax = fs.figure(width=fs.COL_W, height=2.2)
    ...
    fs.finish(fig, "figures/whatever.png")
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ---------------------------------------------------------------- palette --
# Categorical slots, fixed order, never cycled.
BLUE    = "#2a78d6"   # slot 1 -- the "before" / baseline / lab series
ORANGE  = "#eb6834"   # slot 2 -- the "after" / intervention / field series
AQUA    = "#1baf7a"   # slot 3
YELLOW  = "#eda100"   # slot 4
VIOLET  = "#4a3aa7"   # slot 7
RED     = "#e34948"   # slot 8

# Chrome & ink.
INK        = "#0b0b0b"   # primary text
INK_2      = "#52514e"   # secondary text
MUTED      = "#898781"   # axis labels, tick marks
GRID       = "#e1e0d9"   # hairline gridlines
AXIS       = "#c3c2b7"   # baseline / spine
NEUTRAL    = "#b8b6ae"   # "chance" / reference bars -- deliberately colourless
SURFACE    = "#ffffff"   # print surface

# Sequential blue ramp (for magnitude encodings, e.g. confusion matrices).
SEQ_BLUE = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
            "#256abf", "#184f95", "#0d366b"]

# Semantic roles used across the paper, so the same idea is the same colour
# in every figure the reader meets it in.
C_LAB     = BLUE      # laboratory / in-domain
C_FIELD   = ORANGE    # field / out-of-domain
C_BEFORE  = "#9aa3ad"  # baseline / "before" state
C_AFTER   = BLUE      # intervention / "after" state
C_CHANCE  = NEUTRAL   # the chance reference level
C_ATTN    = ORANGE    # measured attention

# ------------------------------------------------------------------ sizes --
# Figures are drawn on a 7.16in canvas and placed at \textwidth, which in
# SN Computer Science's single-column layout is 372pt = 5.15in. Everything
# therefore prints at 5.15/7.16 = 0.719 of its drawn size. Type is specified
# below at the size it should PRINT at, and scaled up by the reciprocal of
# that factor, so an 8pt axis label is an 8pt axis label on the page rather
# than the 5.75pt it would otherwise become. Line weights get the same
# treatment, or hairlines thin out to invisibility in print.
COL_W = 3.45
FULL_W = 7.16
PRINT_W = 5.15                  # \textwidth of the journal's single column
SCALE = FULL_W / PRINT_W        # 1.390

# Sizes as they should appear on the printed page.
FS_TITLE = 8.5 * SCALE
FS_LABEL = 8 * SCALE
FS_TICK = 7.5 * SCALE
FS_ANNOT = 7 * SCALE
FS_LEGEND = 7 * SCALE


def apply():
    """Install the shared rcParams. Call once at the top of a figure script."""
    plt.rcParams.update({
        # Springer asks for 600 dpi on combination art (chart + lettering)
        # and 300 dpi on halftones, measured at final printed size. These
        # are set at final size because the figures are drawn at the width
        # they are placed at; see the note on SCALE above.
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,

        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": FS_LABEL,

        "axes.titlesize": FS_TITLE,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlepad": 6,
        "axes.labelsize": FS_LABEL,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.7 * SCALE,
        "axes.grid": True,
        "axes.axisbelow": True,          # grid never sits on top of the data
        "axes.spines.top": False,
        "axes.spines.right": False,

        "grid.color": GRID,
        "grid.linewidth": 0.6 * SCALE,
        "grid.alpha": 1.0,

        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "xtick.major.width": 0.7 * SCALE,
        "ytick.major.width": 0.7 * SCALE,
        "xtick.major.size": 2.5 * SCALE,
        "ytick.major.size": 2.5 * SCALE,

        "legend.fontsize": FS_LEGEND,
        "legend.frameon": False,
        "legend.handlelength": 1.1,
        "legend.handleheight": 0.9,
        "legend.borderpad": 0.2,
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.35,

        "lines.linewidth": 1.6 * SCALE,
        "lines.markersize": 4 * SCALE,
        "patch.linewidth": 0,
    })


def figure(width=COL_W, height=2.2, **kw):
    return plt.subplots(figsize=(width, height), **kw)


def grid_y_only(ax):
    """Horizontal gridlines only -- the default for bar/category charts."""
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")


def tick_labels(ax, which="both"):
    """Recessive ticks: no tick marks, just labels."""
    ax.tick_params(axis=which, length=0, pad=3)


def panel_title(ax, letter, text):
    """Consistent '(a) Title' panel headings, left-aligned over the plot."""
    ax.set_title(f"({letter})  {text}", loc="left", fontsize=FS_TITLE,
                 fontweight="bold", color=INK, pad=6)


def bar_labels(ax, bars, fmt="{:.1f}", dy=None, color=INK, fontsize=FS_ANNOT,
               weight="normal"):
    """
    Direct value labels above bars. These are not decoration: two palette
    slots sit below 3:1 contrast on white, so the labels are what keeps the
    chart readable in greyscale and for colour-blind readers.
    """
    if dy is None:
        ymin, ymax = ax.get_ylim()
        dy = (ymax - ymin) * 0.02
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize, color=color,
                fontweight=weight)


def headroom(ax, factor=1.18, floor=None):
    """Leave room above the tallest mark so value labels never collide."""
    ymin, ymax = ax.get_ylim()
    top = ymax * factor
    if floor is not None:
        top = max(top, floor)
    ax.set_ylim(ymin if ymin < 0 else 0, top)


def rounded_box(ax, x, y, w, h, facecolor, edgecolor="none", radius=0.018,
                lw=0, zorder=2, alpha=1.0):
    """A rounded node box in axes coordinates -- the diagram primitive."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
        zorder=zorder, alpha=alpha, transform=ax.transAxes,
        mutation_aspect=1.0,
    )
    ax.add_patch(box)
    return box


def arrow(ax, start, end, color=None, lw=1.1, rad=0.0, zorder=3,
          style="-|>", ls="solid", mutation_scale=8):
    """A connector between two points in axes coordinates."""
    color = color or INK_2
    a = FancyArrowPatch(
        start, end, transform=ax.transAxes,
        arrowstyle=style, mutation_scale=mutation_scale,
        connectionstyle=f"arc3,rad={rad}",
        color=color, linewidth=lw, linestyle=ls,
        zorder=zorder, shrinkA=0, shrinkB=0,
        joinstyle="round", capstyle="round",
    )
    ax.add_patch(a)
    return a


def blank_axes(ax, fill=True):
    """
    A pure drawing surface: no ticks, no spines, no grid, 0-1 coordinates.

    `fill` makes the axes occupy the entire figure. This matters for
    diagrams: node geometry is in axes coordinates but text is sized in
    absolute points, so if the axes only covers part of the canvas the two
    scale differently and labels overflow their boxes.
    """
    if fill:
        ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)
    return ax


def finish(fig, path, pad=0.02, tight=True, dpi=None):
    """dpi: leave None for combination art (600); pass 300 for photo grids,
    where the source images carry no detail above that anyway."""
    if tight:
        fig.tight_layout(pad=0.35)
    fig.savefig(path, bbox_inches="tight", pad_inches=pad,
                facecolor=SURFACE, edgecolor="none", dpi=dpi)
    plt.close(fig)
    try:
        from PIL import Image
        im = Image.open(path)
        print(f"  saved {path}  {im.size[0]}x{im.size[1]}px  aspect {im.size[0]/im.size[1]:.3f}")
    except Exception:
        print(f"  saved {path}")
