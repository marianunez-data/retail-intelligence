"""One visual language for the notebook figures (matplotlib)."""
import matplotlib.pyplot as plt

from retail_intelligence.config import FIGURES

BLUE, RED, GRAY, INK, MUTED = "#2a78d6", "#e34948", "#9aa0a6", "#1f2328", "#5f6368"
LIGHT_BLUE = "#a9c8ee"
STATUS = {"Loyal": "#0ca30c", "Nurture": "#fab219", "At risk": "#d03b3b"}
# Fixed categorical order for up to eight series
CATEGORICAL = ["#2a78d6", "#e8743b", "#19a979", "#945ecf", "#ed4a7b", "#13a4b4", "#bf8b2e", "#6c8893"]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "figure.figsize": (9, 4.2),
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c9ccd1",
        "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
        "legend.frameon": False,
    })
    FIGURES.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    """Save into reports/figures and show inline."""
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.show()
