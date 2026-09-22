"""Reading a subsample against a model-card number.

Thinking mode was affordable on 150 CV-Bench items and 100 MathVista items only, drawn
uniformly at random (`rng.sample`, seed 0) from the full sets. Two consequences that pull
in opposite directions and are easy to confuse:

  * the subsample accuracy is an **unbiased** estimate of the full-set accuracy. "82.0% on
    100 items contains 84.50" is a valid statement, not a sleight of hand.
  * this particular draw is **easy**: under thinking off, the protocol both arms share, the
    100 MathVista items score 5.33 points above the other 900. That is well inside sampling
    noise for n = 100 -- a random draw does that -- but it is information, and ignoring it
    leaves the estimate less precise than it has to be.

The standard way to use it is an auxiliary variable: thinking *off* was measured on every
item of both full sets, so for each subsample item we know both arms, and for every item
outside it we know one. Four estimators of the full-set thinking-on accuracy follow, and
the point of printing all four is that the conclusion should not depend on which is chosen:

  direct          the subsample mean. Unbiased, ignores the auxiliary variable.
  difference      full-set off + the paired gain. This is the regression estimator with the
                  coefficient forced to 1, which is only the right coefficient when the two
                  arms are strongly correlated. Here they are not (rho ~ 0.5), so it is the
                  least precise of the four and the furthest from the others.
  regression      full-set off used with the least-squares coefficient. Minimum variance of
                  the four; this is the one to read if only one is read.
  post-stratified reweight the observed fix/break rates by the full set's mix of items the
                  thinking-off arm got right and wrong.

The full-set thinking-off accuracy is a census of the population being estimated (all 2,638
and all 1,000 items), so it carries no sampling error of its own, and the finite-population
correction 1 - n/N applies to the subsample.

    python scripts/representativeness.py        # writes results/representativeness.json

Standard library only: this runs in CI with no dependencies.
"""

import json
import math
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# (name, thinking-on records, thinking-off records over the full set, correctness field, card)
BENCHMARKS = [
    (
        "cvbench",
        "cvbench_think_on_sub150.jsonl",
        "cvbench_think_off.jsonl",
        "ok",
        86.82,
    ),
    (
        "mathvista",
        "mathvista_think_on_sub100_llmext.json",
        "mathvista_think_off_b2048_llmext.json",
        "llm_ok",
        84.50,
    ),
]
BOOTSTRAP = 4000


def load(name):
    p = RESULTS / name
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return json.loads(p.read_text())


def wilson(k, n, z=1.96):
    p = k / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (centre - half) / (1 + z * z / n), 100 * (centre + half) / (
        1 + z * z / n
    )


def two_proportion_p(k1, n1, k2, n2):
    """Two-sided z test for subsample-vs-rest, pooled variance."""
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    return math.erfc(abs(k1 / n1 - k2 / n2) / se / math.sqrt(2))


def mcnemar_exact(pairs):
    b = sum(1 for on, off in pairs if on and not off)
    c = sum(1 for on, off in pairs if off and not on)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    return b, c, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def _moments(y, x):
    n = len(y)
    my, mx = sum(y) / n, sum(x) / n
    sy = sum((t - my) ** 2 for t in y) / (n - 1)
    sx = sum((t - mx) ** 2 for t in x) / (n - 1)
    sxy = sum((a - my) * (b - mx) for a, b in zip(y, x)) / (n - 1)
    return my, mx, sy, sx, sxy


def estimators(y, x, A, N):
    """Four estimates of the full-set thinking-on accuracy, as (value, standard error)."""
    n = len(y)
    my, mx, sy, sx, sxy = _moments(y, x)
    fpc = 1 - n / N
    b = sxy / sx
    rho = sxy / math.sqrt(sy * sx)
    d = [a - c for a, c in zip(y, x)]
    md = sum(d) / n
    sd = sum((t - md) ** 2 for t in d) / (n - 1)

    hard = [a for a, c in zip(y, x) if c == 0]
    easy = [a for a, c in zip(y, x) if c == 1]
    r_fix = sum(hard) / len(hard)
    r_break = 1 - sum(easy) / len(easy)
    p_hard = 1 - A
    strat = A + p_hard * r_fix - (1 - p_hard) * r_break
    v_strat = p_hard**2 * r_fix * (1 - r_fix) / len(hard) + (
        1 - p_hard
    ) ** 2 * r_break * (1 - r_break) / len(easy)

    return (
        {
            "direct": (my, math.sqrt(fpc * sy / n)),
            "difference": (A + md, math.sqrt(fpc * sd / n)),
            "regression": (my + b * (A - mx), math.sqrt(fpc * sy * (1 - rho**2) / n)),
            "post_stratified": (strat, math.sqrt(v_strat)),
        },
        b,
        rho,
        r_fix,
        r_break,
    )


def bootstrap_ci(y, x, A, N, seed=0, draws=BOOTSTRAP):
    """Percentile intervals, resampling the subsample. Cross-check on the analytic errors."""
    rng = random.Random(seed)
    n = len(y)
    keep = {k: [] for k in ("direct", "difference", "regression", "post_stratified")}
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        by, bx = [y[i] for i in idx], [x[i] for i in idx]
        my, mx, _, sx, sxy = _moments(by, bx)
        keep["direct"].append(my)
        keep["difference"].append(A + my - mx)
        keep["regression"].append(my + (sxy / sx if sx else 0.0) * (A - mx))
        hard = [a for a, c in zip(by, bx) if c == 0]
        easy = [a for a, c in zip(by, bx) if c == 1]
        if hard and easy:
            keep["post_stratified"].append(
                A + (1 - A) * (sum(hard) / len(hard)) - A * (1 - sum(easy) / len(easy))
            )
    out = {}
    for k, vals in keep.items():
        vals.sort()
        out[k] = [
            round(100 * vals[int(0.025 * len(vals))], 1),
            round(100 * vals[int(0.975 * len(vals))], 1),
        ]
    return out


def compute():
    out = {}
    for name, on_file, off_file, key, card in BENCHMARKS:
        on = {r["id"]: float(r[key]) for r in load(on_file)}
        off = {r["id"]: float(r[key]) for r in load(off_file)}
        subset = [i for i in off if i in on]
        rest = [i for i in off if i not in on]
        n, N = len(subset), len(off)

        y = [on[i] for i in subset]
        x = [off[i] for i in subset]
        A = sum(off.values()) / N

        k_sub, k_rest = sum(x), sum(off[i] for i in rest)
        pairs = [(on[i], off[i]) for i in subset]
        b_mc, c_mc, p_mc = mcnemar_exact(pairs)

        est, b, rho, r_fix, r_break = estimators(y, x, A, N)
        boot = bootstrap_ci(y, x, A, N)

        rows = {}
        for k, (v, se) in est.items():
            rows[k] = {
                "value": round(100 * v, 2),
                "se": round(100 * se, 2),
                "ci": [
                    round(100 * (v - 1.96 * se), 1),
                    round(100 * (v + 1.96 * se), 1),
                ],
                "bootstrap_ci": boot[k],
                "contains_card": 100 * (v - 1.96 * se) <= card <= 100 * (v + 1.96 * se),
            }
        shortfalls = sorted(round(card - r["value"], 2) for r in rows.values())

        out[name] = {
            "card": card,
            "n_subset": n,
            "n_rest": len(rest),
            "n_full": N,
            "full_off": round(100 * A, 2),
            "full_off_ci": [round(v, 1) for v in wilson(round(100 * A * N / 100), N)],
            "subset_off": round(100 * k_sub / n, 2),
            "rest_off": round(100 * k_rest / len(rest), 2),
            "subset_easier_by": round(100 * (k_sub / n - k_rest / len(rest)), 2),
            "representativeness_p": round(
                two_proportion_p(k_sub, n, k_rest, len(rest)), 2
            ),
            "subset_on": round(100 * sum(y) / n, 2),
            "correlation": round(rho, 3),
            "regression_coefficient": round(b, 3),
            "fix_rate_on_items_off_got_wrong": round(100 * r_fix, 2),
            "break_rate_on_items_off_got_right": round(100 * r_break, 2),
            "paired_gain": round(100 * (sum(y) - sum(x)) / n, 2),
            "mcnemar_b": b_mc,
            "mcnemar_c": c_mc,
            "mcnemar_p": round(p_mc, 3),
            "estimators": rows,
            "shortfall_min": shortfalls[0],
            "shortfall_max": shortfalls[-1],
            "every_interval_contains_card": all(
                r["contains_card"] for r in rows.values()
            ),
        }
    return out


def main() -> int:
    out = compute()
    (RESULTS / "representativeness.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n"
    )
    for name, r in out.items():
        print(
            f"  {name}  (subsample {r['n_subset']} of {r['n_full']}, card {r['card']})"
        )
        print(
            f"    full set off {r['full_off']:.2f}%   subsample off {r['subset_off']:.2f}%   "
            f"rest {r['rest_off']:.2f}%   easier by {r['subset_easier_by']:+.2f}, "
            f"p = {r['representativeness_p']}"
        )
        print(
            f"    rho = {r['correlation']}   least-squares coefficient = "
            f"{r['regression_coefficient']}"
        )
        for k, e in r["estimators"].items():
            print(
                f"      {k:<16}{e['value']:>7.2f}%  se {e['se']:>4.2f}  "
                f"CI {str(e['ci']):>16}  bootstrap {str(e['bootstrap_ci']):>16}  "
                f"{'contains the card' if e['contains_card'] else 'EXCLUDES THE CARD'}"
            )
        print(
            f"    every estimate is {r['shortfall_min']:+.2f} to {r['shortfall_max']:+.2f} "
            f"from the card; every interval contains it: "
            f"{r['every_interval_contains_card']}"
        )
    print("  wrote results/representativeness.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
