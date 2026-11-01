"""Generate the study-flow diagram used in the PCS slide.

The staff example (slides_assets/enrichment_example.png) is a 300-dpi
render of Thesis_SCL/figures/fig_enrichment.pdf, cropped to content.
"""
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(_HERE, "slides_assets")
os.makedirs(OUT, exist_ok=True)

RED = "#E30520"
TXT = "#333333"
FILL = "#F4F4F4"
EDGE = "#C9C9C9"

steps = [
    "53 listeners",
    "43 valid",
    "1,032 pairs",
    "11 properties",
    "4 kept",
    "Score 0–1",
]

fig_w, fig_h = 8.8, 0.82
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")

n = len(steps)
box_w = 1.34
gap = (fig_w - n * box_w) / (n - 1)
y0, box_h = 0.22, 0.40

for i, label in enumerate(steps):
    x = i * (box_w + gap)
    last = i == n - 1
    ax.add_patch(FancyBboxPatch(
        (x, y0), box_w, box_h,
        boxstyle="round,pad=0.01,rounding_size=0.07",
        linewidth=1.2, edgecolor=RED if last else EDGE,
        facecolor="#FDEDEF" if last else FILL, zorder=2))
    ax.text(x + box_w / 2, y0 + box_h / 2, label,
            ha="center", va="center", fontsize=10,
            color=RED if last else TXT, zorder=3)
    if not last:
        ax.add_patch(FancyArrowPatch(
            (x + box_w + 0.03, y0 + box_h / 2),
            (x + box_w + gap - 0.03, y0 + box_h / 2),
            arrowstyle="-|>", mutation_scale=11, linewidth=1.3,
            color=RED, zorder=1))

fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(os.path.join(OUT, "study_flow.png"), dpi=300,
            facecolor="white")
print("wrote", os.path.join(OUT, "study_flow.png"))
