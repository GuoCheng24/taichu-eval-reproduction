"""Generate the GitHub social-preview card (1200x630). Reproducible: python3 make_social_preview.py

The finding is a direction: on CV-Bench every estimator of the full-set accuracy lands above the
card's number, on MathVista every one lands below it. So the card draws the span the four
estimators cover against the card value, on a shared scale, and the two spans fall on opposite
sides of their marks.

Numbers come from results/representativeness.json, the file scripts/check_readme_numbers.py holds
the page to cell by cell.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cardkit import SANS, card  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = json.loads((ROOT / "results/representativeness.json").read_text())


def values(key):
    return sorted(e["value"] for e in D[key]["estimators"].values()), D[key]["card"]


ROWS = [("CV-Bench", *values("cvbench")), ("MathVista", *values("mathvista"))]
LO, HI = -8.6, 6.4                       # points away from the card number, shared axis


def chart(ax, accent):
    """Four dots per benchmark, against the card number.

    Drawn as a span bar first: CV-Bench's four estimators cover 0.88 points against
    MathVista's 4.80, so on a shared axis the green one rendered as a mark too small to see.
    Dots do not shrink with the span, and four of them is also what the sentence says.
    """
    x0, x1 = 3.95, 10.35

    def X(v):
        return x0 + (x1 - x0) * (v - LO) / (HI - LO)

    # the line stops clear of its own label: a vertical rule under text
    # passed every check until the rule test learned about vertical rules
    ax.plot([X(0), X(0)], [1.05, 3.34], color="#17181a", lw=3, zorder=2)
    ax.text(X(0), 3.62, "the card number", fontsize=34, color="#17181a", family=SANS,
            ha="center")
    for i, (name, vals, cardv) in enumerate(ROWS):
        y = 2.80 - i * 1.12
        above = min(vals) > cardv
        colour = "#1a7f37" if above else "#bc4c00"
        d = [v - cardv for v in vals]
        ax.plot([X(min(d)), X(max(d))], [y, y], color=colour, lw=3, zorder=3, alpha=0.55)
        ax.plot([X(v) for v in d], [y] * len(d), "o", ms=17, color=colour, zorder=4)
        ax.text(x0 - 0.24, y, name, fontsize=34, color="#17181a", family=SANS,
                ha="right", va="center")
        label = f"+{min(d):.1f} to +{max(d):.1f}" if above else f"{min(d):.1f} to {max(d):.1f}"
        ax.text(X(max(max(d), 0)) + 0.30, y, label, fontsize=34, fontweight="bold",
                color=colour, family=SANS, va="center", ha="left")


out = card(
    out=str(pathlib.Path(__file__).parent / "social-preview.png"),
    accent="#8250df", badge="Z",
    kicker="REPRODUCTION  ·  ZDTaichu5.0-9B on one RTX 4090",
    headline="One card number above, one below",
    evidence="how far four estimators land from the card",
    chart=chart,
    footer="github.com/GuoCheng24/taichu-eval-reproduction",
    headline_size=44,
)
print(f"written {pathlib.Path(out).name}  "
      f"CV {min(ROWS[0][1]):.2f}-{max(ROWS[0][1]):.2f} vs {ROWS[0][2]}  "
      f"MV {min(ROWS[1][1]):.2f}-{max(ROWS[1][1]):.2f} vs {ROWS[1][2]}")
