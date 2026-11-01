"""Generate the PCS funnel, chord tensor and overlap-vs-key scatter."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
_HERE = os.path.dirname(os.path.abspath(__file__))
for _f in ("Montserrat-Regular.ttf", "Montserrat-Bold.ttf",
           "Montserrat-Italic.ttf", "Montserrat-SemiBold.ttf"):
    fm.fontManager.addfont(os.path.join(_HERE, "fonts", _f))
plt.rcParams["font.family"] = "Montserrat"
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle

OUT = os.path.join(_HERE, "slides_assets")
RED = "#E30520"
DK = "#595959"
GREY = "#808080"
DARK_RED = "#B00418"
ORANGE = "#FF9900"
GRAY = "#808080"
RED = "#E30520"


def rbox(ax, x, y, w, h, fc, ec, lw=1.0, rs=0.06):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.01,rounding_size={rs}",
                                fc=fc, ec=ec, lw=lw, zorder=3))


# ---------------------------------------------------------------- funnel
W, H = 4.6, 2.7
fig, ax = plt.subplots(figsize=(W, H), dpi=300)
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

rbox(ax, 0.55, 2.28, 3.50, 0.32, "#FFFFFF", "#BDBDBD", lw=1.2)
ax.text(2.30, 2.44, "11 properties measured with jSymbolic", ha="center",
        va="center", fontsize=8, color=DK, fontweight="bold")

funnel = Polygon([(0.75, 2.22), (3.85, 2.22), (3.05, 0.92), (1.55, 0.92)],
                 closed=True, fc="#EDEDED", ec="#C4C4C4", lw=1.1, zorder=2)
ax.add_patch(funnel)

for y, txt in ((1.86, "Human filter: 43 listeners, GMSI groups"),
               (1.52, "Statistical filter: p < 0.05, same sign")):
    ax.plot([0.95, 3.65], [y, y], color="#9A9A9A", lw=0.9, ls=(0, (4, 2)),
            zorder=4)
    ax.text(2.30, y + 0.05, txt, ha="center", va="bottom", fontsize=6.8,
            color=DK, zorder=5,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                      alpha=0.9))

chips = [("7C", RED, "white"), ("VNSPC", DARK_RED, "white"),
         ("DTMCVI", ORANGE, "#2B2B2B"), ("VDR", GRAY, "white")]
cw, gap = 0.72, 0.10
x0 = (W - (4 * cw + 3 * gap)) / 2
for i, (lab, col, tc) in enumerate(chips):
    x = x0 + i * (cw + gap)
    rbox(ax, x, 0.30, cw, 0.40, col, col, lw=0, rs=0.05)
    ax.text(x + cw / 2, 0.50, lab, ha="center", va="center", fontsize=7,
            color=tc, fontweight="bold")

fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(os.path.join(OUT, "pcs_funnel.png"), dpi=300, facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- tensor
W, H = 8.6, 3.0
fig, ax = plt.subplots(figsize=(W, H), dpi=300)
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

blocks = [("Root", 12, RED, "white", [0]),
          ("Quality", 8, DARK_RED, "white", [0]),
          ("7th", 5, ORANGE, "#2B2B2B", [2]),
          ("Ext", 3, GRAY, "white", []),
          ("Alt", 4, RED, "white", []),
          ("Add", 3, DARK_RED, "white", []),
          ("Bass", 13, GRAY, "white", [12])]
total = sum(b[1] for b in blocks)
x, y, h, dx, dy = 0.35, 1.55, 1.00, 0.10, 0.10
for name, n, col, tc, active in blocks:
    w = (8.6 - 0.75) * n / total
    ax.add_patch(Rectangle((x, y), w, h, fc=col, ec="#222222", lw=0.8,
                           zorder=3))
    ax.add_patch(Polygon([(x, y + h), (x + dx, y + h + dy),
                          (x + w + dx, y + h + dy), (x + w, y + h)],
                         closed=True, fc=col, ec="#222222", lw=0.8,
                         alpha=0.92, zorder=2))
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy),
                          (x + w + dx, y + h + dy), (x + w, y + h)],
                         closed=True, fc=col, ec="#222222", lw=0.8,
                         alpha=0.75, zorder=2))
    ax.text(x + w / 2, y + h * 0.62, name, ha="center", va="center",
            fontsize=7.5, color=tc, fontweight="bold", zorder=4)
    ax.text(x + w / 2, y + h * 0.28, str(n), ha="center", va="center",
            fontsize=7.5, color=tc, zorder=4)
    # bit row for this slot
    size = min(0.13, w / (n + 1.2))
    gap = (w - n * size) / (n + 1)
    for i in range(n):
        bx = x + gap + i * (size + gap)
        on = i in active
        ax.add_patch(Rectangle((bx, 0.82), size, size,
                               fc=RED if on else "#EDEDED",
                               ec="#C9C9C9" if not on else RED, lw=0.6,
                               zorder=4))
    x += w

ax.annotate("", xy=(8.25, 2.85), xytext=(0.35, 2.85),
            arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.9))
ax.text(4.30, 2.90, "48 values per chord", ha="center", va="bottom",
        fontsize=8, color=GREY)
ax.text(0.35, 1.20, "Example: Cmaj7", fontsize=8.5, color=DK,
        fontweight="bold")
ax.text(0.35, 0.20, "4 active values: root C  ·  quality major  ·  "
        "seventh maj7  ·  bass root", fontsize=7.5, color=GREY)

fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(os.path.join(OUT, "chord_tensor.png"), dpi=300, facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- scatter
W, H = 4.5, 3.0
fig, ax = plt.subplots(figsize=(W, H), dpi=300)
pts = [("CVAE", 84, 84, "#808080", "s", 8, (6, 8)),
       ("RVAE", 76, 64, "#595959", "D", 7, (8, 6)),
       ("PEARL-v1", 88, 32, RED, "o", 0, (-56, -4)),
       ("PEARL-v2", 88, 36, ORANGE, "o", 0, (10, 8))]
for name, xv, yv, col, mk, _, off in pts:
    ax.scatter([xv], [yv], s=110 if name.startswith("PEARL") else 70,
               color=col, marker=mk, zorder=4,
               edgecolor="white" if name.startswith("PEARL") else "none")
    ax.annotate(name, (xv, yv), xytext=off, textcoords="offset points",
                fontsize=8, color=col if name.startswith("PEARL") else DK,
                fontweight="bold" if name.startswith("PEARL") else "normal")

ax.set_xlim(72, 92); ax.set_ylim(25, 92)
ax.set_xlabel("Scale adherence (%)", fontsize=8.5, color=DK)
ax.set_ylabel("Overlap with input (%)", fontsize=8.5, color=DK)
ax.tick_params(labelsize=7.5, colors=GREY)
for sp in ax.spines.values():
    sp.set_color("#C9C9C9")
ax.grid(color="#EDEDED", lw=0.8, zorder=0)

fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "overlap_vs_key.png"), dpi=300,
            facecolor="white")
plt.close(fig)
print("ok")

# ------------------------------------------------------- latent control
from PIL import Image
import numpy as np

src = os.path.join(OUT, "pca_colored.png")
im = Image.open(src).crop((0, 0, 845, 852))
W, H = 4.2, 3.2
fig, ax = plt.subplots(figsize=(W, H), dpi=300)
ax.imshow(np.asarray(im))
ax.axis("off")

start, end = (0.31, 0.55), (0.80, 0.55)
ax.annotate("", xy=end, xytext=start, xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.0,
                            mutation_scale=16, shrinkA=6, shrinkB=6))
ax.scatter([start[0]], [start[1]], transform=ax.transAxes, s=70,
           color="#808080", zorder=5)
ax.scatter([end[0]], [end[1]], transform=ax.transAxes, s=90, color=RED,
           zorder=5, edgecolor="white", linewidth=1.2)
ax.text(start[0], start[1] + 0.05, "input", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=8.5, color=DK, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                  alpha=0.85))
ax.text(end[0], end[1] + 0.05, "target", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=8.5, color=RED,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                  alpha=0.85))
ax.text((start[0] + end[0]) / 2, start[1] + 0.09, "move z",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
        color=RED, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                  alpha=0.85))

fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(os.path.join(OUT, "latent_control.png"), dpi=300,
            facecolor="white")
plt.close(fig)
print("latent ok")
