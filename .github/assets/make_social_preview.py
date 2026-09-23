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


# CV-Bench is no longer estimated: the full set was generated and scored, so its
# row is the measurement and its exact interval. MathVista's full-set arm is
# still running, so that row is still the four estimators. Showing both as
# estimators would be a card that disagrees with its own README.
FULL_CV = json.loads((ROOT / "results/metrics_fullset_cvbench.json").read_text())
FULL_MV = json.loads((ROOT / "results/metrics_fullset_mathvista.json").read_text())


def measured(d, key="accuracy_pct", ci="ci_pct"):
    """(point, lo, hi) as distances from the card number."""
    c = d["card"]
    return d[key] - c, d[ci][0] - c, d[ci][1] - c


# Three rows, because MathVista has two scorings and they disagree about the
# card: 37 of its 1,000 generations never closed their reasoning block, the
# extractor credits 16 of them, and counting those the other way moves the
# interval off the card line. A card that showed one scoring would be picking
# the verdict.
ROWS = [
    ("CV-Bench", measured(FULL_CV)),
    ("MathVista", measured(FULL_MV)),
    ("strict scoring", measured(FULL_MV, "accuracy_unclosed_as_wrong_pct",
                                    "ci_unclosed_as_wrong_pct")),
]
LO, HI = -7.6, 3.4                       # points away from the card number, shared axis


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
    ax.plot([X(0), X(0)], [0.80, 3.30], color="#17181a", lw=3, zorder=2)
    ax.text(X(0), 3.56, "the card number", fontsize=34, color="#17181a", family=SANS,
            ha="center")
    for i, (name, (point, lo, hi)) in enumerate(ROWS):
        y = 2.92 - i * 0.92
        # green when the interval still covers the card number, orange when it
        # does not - the only distinction this card is making
        colour = "#1a7f37" if lo <= 0 <= hi else "#bc4c00"
        ax.text(x0 - 0.24, y, name, fontsize=34, color="#17181a", family=SANS,
                ha="right", va="center")
        ax.plot([X(lo), X(hi)], [y, y], color=colour, lw=3, zorder=3, alpha=0.55)
        ax.plot([X(point)], [y], "o", ms=20, color=colour, zorder=4)
        ax.text(X(max(hi, 0)) + 0.26, y, f"{point:+.2f}", fontsize=34,
                fontweight="bold", color=colour, family=SANS, va="center", ha="left")


out = card(
    out=str(pathlib.Path(__file__).parent / "social-preview.png"),
    accent="#8250df", badge="Z",
    kicker="REPRODUCTION  ·  ZDTaichu5.0-9B",
    headline="One verdict rests on 37 truncations",
    evidence="measured minus the card, exact 95% intervals",
    chart=chart,
    footer="github.com/GuoCheng24/taichu-eval-reproduction",
    headline_size=44,
)
print("written " + pathlib.Path(out).name + "  "
      + "  ".join(f"{n.strip()} {p:+.2f} [{lo:+.2f},{hi:+.2f}]" for n, (p, lo, hi) in ROWS))
