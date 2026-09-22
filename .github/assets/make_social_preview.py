"""Generate the GitHub social-preview card (1280x640). Reproducible: python3 make_social_preview.py

Numbers come from results/representativeness.json, the file scripts/check_readme_numbers.py holds
the page to, so the card and the page cannot disagree.
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
d = json.loads((ROOT / "results/representativeness.json").read_text())
cv, mv = d["cvbench"], d["mathvista"]


def span(r):
    v = [e["value"] for e in r["estimators"].values()]
    return min(v), max(v)


cv_lo, cv_hi = span(cv)
mv_lo, mv_hi = span(mv)

W, H = 12.8, 6.4
fig = plt.figure(figsize=(W, H), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
ax.add_patch(plt.Rectangle((0, 0), W, H, color="#0d1117"))
SANS, MONO = "Liberation Sans", "Liberation Mono"

ax.text(0.75, 5.55, "taichu-eval-reproduction", fontsize=32, fontweight="bold", color="#e6edf3", family=SANS)
ax.text(0.75, 4.92, "Two model-card numbers re-measured on one RTX 4090, and how to read a subsample against them.",
        fontsize=16, color="#8b949e", family=SANS)

ax.add_patch(FancyBboxPatch((0.72, 1.28), 11.36, 3.05, boxstyle="round,pad=0.12",
                            fc="#161b22", ec="#30363d", lw=1.5))
ax.text(0.95, 4.02, "$ python scripts/representativeness.py      # ZDTaichu5.0-9B, card protocol",
        fontsize=13.5, color="#7d8590", family=MONO)
rows = [
    (f"CV-Bench    {cv_lo:.2f} - {cv_hi:.2f}%   vs card {cv['card']}   every estimate ABOVE it", "#3fb950"),
    (f"MathVista   {mv_lo:.2f} - {mv_hi:.2f}%   vs card {mv['card']}   every estimate below it", "#f0883e"),
    (f"the catch   the {mv['n_subset']} items thinking mode could afford run {mv['subset_easier_by']:.2f}", "#58a6ff"),
    (f"            points easier than the other {mv['n_rest']} - so four estimators, not one", "#58a6ff"),
]
y = 3.55
for txt, c in rows:
    ax.text(0.95, y, txt, fontsize=14, color=c, family=MONO)
    y -= 0.5
ax.text(0.95, y - 0.02,
        "direct | difference | regression | post-stratified - and the card is inside all eight intervals",
        fontsize=12, color="#7d8590", family=MONO)

ax.text(0.75, 0.62,
        "A uniform subsample mean is unbiased; it is also imprecise, and this draw was easy. "
        "Both facts are on the page.",
        fontsize=12.5, color="#8b949e", family=SANS)

out = pathlib.Path(__file__).parent / "social-preview.png"
fig.savefig(out)
print(f"written {out.name} 1280x640 (CV {cv_lo:.2f}-{cv_hi:.2f}, MV {mv_lo:.2f}-{mv_hi:.2f})")
