"""Generate the pipeline, per-property and target-gauge diagrams."""
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = os.path.join(_HERE, "slides_assets")
RED = "#E30520"
DK = "#595959"
GREY = "#808080"
BOX_FC = "#F6F6F6"
BOX_EC = "#C9C9C9"


def rbox(ax, x, y, w, h, fc=BOX_FC, ec=BOX_EC, lw=1.1, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.01,rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(ax, x1, y1, x2, y2, color=RED, lw=1.4, size=12):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=size, lw=lw, color=color,
                                 zorder=5, shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------- pipeline
W, H = 8.8, 2.05
fig, ax = plt.subplots(figsize=(W, H), dpi=300)
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

def chord_row(x0, y, n, notes, color):
    bw, gap = 0.26, 0.06
    for i in range(n):
        x = x0 + i * (bw + gap)
        rbox(ax, x, y, bw, 0.46)
        cx = x + bw / 2
        for k in range(notes):
            ax.add_patch(Circle((cx, y + 0.09 + k * 0.09), 0.026,
                                color=color, zorder=4))

chord_row(0.10, 0.86, 5, 3, GREY)
ax.text(0.10, 1.42, "progression", fontsize=9.5, color=DK, fontweight="bold")
arrow(ax, 1.72, 1.09, 2.05, 1.09)

rbox(ax, 2.08, 0.66, 0.95, 0.86, fc="#FDF3F4", ec=RED)
ax.text(2.555, 1.09, "Encoder", ha="center", va="center", fontsize=9.5,
        color=DK, fontweight="bold")
arrow(ax, 3.05, 1.09, 3.36, 1.09)

rbox(ax, 3.39, 0.52, 1.55, 1.14)
ax.text(4.165, 1.50, "z  (32 numbers)", ha="center", va="center",
        fontsize=9.5, color=DK, fontweight="bold")
for r in range(4):
    for c in range(8):
        ax.add_patch(Circle((3.62 + c * 0.17, 1.20 - r * 0.17), 0.028,
                            color="#FF9900", zorder=4))
arrow(ax, 4.96, 1.09, 5.29, 1.09)

rbox(ax, 5.32, 0.66, 0.95, 0.86, fc="#FDF3F4", ec=RED)
ax.text(5.795, 1.09, "Decoder", ha="center", va="center", fontsize=9.5,
        color=DK, fontweight="bold")
arrow(ax, 6.29, 1.09, 6.62, 1.09)

chord_row(6.65, 0.86, 5, 4, RED)
ax.text(6.65, 1.42, "new progression", fontsize=9.5, color=DK,
        fontweight="bold")

# predictors under z
rbox(ax, 3.42, 0.10, 0.74, 0.30, fc="#FFFFFF")
ax.text(3.79, 0.25, "complexity", ha="center", va="center", fontsize=8,
        color=DK)
rbox(ax, 4.24, 0.10, 0.50, 0.30, fc="#FFFFFF")
ax.text(4.49, 0.25, "key", ha="center", va="center", fontsize=8, color=DK)
arrow(ax, 3.80, 0.52, 3.80, 0.42, color=GREY, lw=1.0, size=8)
arrow(ax, 4.47, 0.52, 4.47, 0.42, color=GREY, lw=1.0, size=8)

fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(os.path.join(OUT, "pearl_pipeline.png"), dpi=300,
            facecolor="white")
plt.close(fig)

# ------------------------------------------------------- per-property cards
W, H = 8.8, 1.62
fig, ax = plt.subplots(figsize=(W, H), dpi=300)
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

def card(x, y, w, h, name, notes, color):
    rbox(ax, x, y, w, h, fc="#FFFFFF", ec=color, lw=1.3)
    ax.text(x + w / 2, y + h * 0.62, name, ha="center", va="center",
            fontsize=10.5, color=DK, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.26, f"{notes} notes", ha="center",
            va="center", fontsize=8.5, color=GREY)

rows = [
    ("Seventh-chord target", RED,
     [("Cmaj7", 4), ("Dmin7", 4), ("Cmin7", 4), ("Gmin7", 4), ("Cmin7", 4)]),
    ("Dissonance target", "#FF9900",
     [("Dmin", 3), ("Amaj", 3), ("Csus4add9", 4), ("Dsus4", 3), ("Amin", 3)]),
]
y_rows = [0.88, 0.10]
for (label, color, chords), y in zip(rows, y_rows):
    ax.text(0.08, y + 0.33, label, fontsize=10, color=color,
            fontweight="bold", va="center")
    x0 = 2.05
    cw, gap = 1.22, 0.10
    for i, (name, n) in enumerate(chords):
        card(x0 + i * (cw + gap), y, cw, 0.62, name, n, color)

fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(os.path.join(OUT, "property_outputs.png"), dpi=300,
            facecolor="white")
plt.close(fig)
print("ok")

# ------------------------------------------------------------------ gauge
W, H = 4.3, 1.05
fig, ax = plt.subplots(figsize=(W, H), dpi=300)
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

y = 0.40
ax.plot([0.15, 4.15], [y, y], color="#D0D0D0", lw=2.2, zorder=1,
        solid_capstyle="round")
ax.scatter([0.90], [y], s=70, color="#8A8A8A", zorder=3)
ax.text(0.90, y + 0.10, "start", ha="center", va="bottom", fontsize=8.5,
        color=DK, fontweight="bold")
ax.scatter([3.50], [y], s=90, color=RED, zorder=3)
ax.text(3.50, y + 0.10, "target", ha="center", va="bottom", fontsize=8.5,
        color=RED, fontweight="bold")
arrow(ax, 1.10, y + 0.36, 3.30, y + 0.36, lw=1.3, size=10)
ax.text(2.20, y + 0.41, "move z: 20–30 steps", ha="center", va="bottom",
        fontsize=8.5, color=DK)
ax.text(2.15, 0.03, "complexity score", ha="center", va="bottom",
        fontsize=8, color=GREY, style="italic")

fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(os.path.join(OUT, "target_gauge.png"), dpi=300,
            facecolor="white")
print("gauge ok")
