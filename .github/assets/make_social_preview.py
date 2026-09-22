"""Generate the GitHub social-preview card (1200x630). Reproducible: python3 make_social_preview.py

The first version put its message in a block of 14 pt monospace. A social card is unfurled at about
360 px wide in Slack, where that is grey noise, so the message is in the headline now and the
terminal panel is texture beside it. Layout in lightcard.py next to this file, which refuses to emit a card whose left column runs under the panel.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from lightcard import draw  # noqa: E402

out = draw(
    out=str(pathlib.Path(__file__).parent / "social-preview.png"),
    accent="#8250df", badge="Z", headline_size=42,
    kicker="REPRODUCTION  ·  ZDTaichu5.0-9B on one RTX 4090",
    headline="One card number above, one below",
    subline="and four estimators, not one",
    body=["Thinking mode was affordable on a",
          "random subsample, and that subsample",
          "runs 5.33 points easier than the rest.",
          "So the estimator is the argument."],
    panel=[("$ python scripts/representativeness.py", "dim"),
           ("CV-Bench   88.45-89.33%  card 86.82", "ok"),
           ("           every estimate above it", "ok"),
           ("MathVista  77.20-82.00%  card 84.50", "warn"),
           ("           every estimate below it", "warn")],
    footer="github.com/GuoCheng24/taichu-eval-reproduction",
)
print(f"written {pathlib.Path(out).name} 1200x630")
