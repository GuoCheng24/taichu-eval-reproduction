"""Is a thinking-on subset as hard as the benchmark it stands for?

Thinking mode was affordable on 150 CV-Bench items and 100 MathVista items only. A subset accuracy
is a statement about the benchmark only if the subset is as hard as the rest of it, and the way to
find out is already on disk: both subsets were also scored under thinking *off*, the setting where
the full set was measured too. Comparing subset-off against the-rest-off answers it directly, with
no model run.

What survives a subset that turns out to be easy is the paired gain, which sees the same items in
both arms, so this also recomputes that gain, its exact McNemar test, and what it implies for the
full set when carried onto the full-set thinking-off accuracy.

The projected interval combines the full-set binomial standard error with the paired-difference
standard error in quadrature. The two are not strictly independent -- the subset is part of the full
set -- but it is 150 of 2,638 and 100 of 1,000, so the overlap term is small.

    python scripts/representativeness.py        # writes results/representativeness.json

Standard library only: this runs in CI with no dependencies.
"""

import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# (name, thinking-on records, thinking-off records over the full set, the field holding correctness)
BENCHMARKS = [
    ("cvbench", "cvbench_think_on_sub150.jsonl", "cvbench_think_off.jsonl", "ok"),
    (
        "mathvista",
        "mathvista_think_on_sub100_llmext.json",
        "mathvista_think_off_b2048_llmext.json",
        "llm_ok",
    ),
]


def load(name):
    p = RESULTS / name
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return json.loads(p.read_text())


def wilson(k, n, z=1.96):
    p = k / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    denom = 1 + z * z / n
    return 100 * (centre - half) / denom, 100 * (centre + half) / denom


def two_proportion_p(k1, n1, k2, n2):
    """Two-sided z test for subset-vs-rest, pooled variance."""
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (k1 / n1 - k2 / n2) / se
    return math.erfc(abs(z) / math.sqrt(2))


def mcnemar_exact(pairs):
    """Exact two-sided binomial test on the discordant pairs."""
    b = sum(1 for on, off in pairs if on and not off)
    c = sum(1 for on, off in pairs if off and not on)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return b, c, min(1.0, p)


def paired_difference(pairs, z=1.96):
    """Mean of the per-item differences, with its standard error (percentage points)."""
    n = len(pairs)
    diffs = [(1 if on else 0) - (1 if off else 0) for on, off in pairs]
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    return 100 * mean, 100 * (mean - z * se), 100 * (mean + z * se), 100 * se


def compute():
    out = {}
    for name, on_file, off_file, key in BENCHMARKS:
        on = {r["id"]: bool(r[key]) for r in load(on_file)}
        off = {r["id"]: bool(r[key]) for r in load(off_file)}
        subset = [i for i in off if i in on]
        rest = [i for i in off if i not in on]

        k_sub, n_sub = sum(off[i] for i in subset), len(subset)
        k_rest, n_rest = sum(off[i] for i in rest), len(rest)
        k_full, n_full = sum(off.values()), len(off)

        pairs = [(on[i], off[i]) for i in subset]
        b, c, p_mcnemar = mcnemar_exact(pairs)
        delta, d_lo, d_hi, d_se = paired_difference(pairs)

        full_off = 100 * k_full / n_full
        projected = full_off + delta
        se_full = 100 * math.sqrt((k_full / n_full) * (1 - k_full / n_full) / n_full)
        se_proj = math.sqrt(se_full**2 + d_se**2)

        out[name] = {
            "n_subset": n_sub,
            "n_rest": n_rest,
            "subset_off": round(100 * k_sub / n_sub, 2),
            "rest_off": round(100 * k_rest / n_rest, 2),
            "subset_easier_by": round(100 * k_sub / n_sub - 100 * k_rest / n_rest, 2),
            "representativeness_p": round(
                two_proportion_p(k_sub, n_sub, k_rest, n_rest), 2
            ),
            "subset_on": round(100 * sum(1 for o, _ in pairs if o) / n_sub, 2),
            "paired_gain": round(delta, 2),
            "paired_gain_ci": [round(d_lo, 2), round(d_hi, 2)],
            "mcnemar_b": b,
            "mcnemar_c": c,
            "mcnemar_p": round(p_mcnemar, 3),
            "full_off": round(full_off, 2),
            "full_off_ci": [round(x, 1) for x in wilson(k_full, n_full)],
            "projected_on": round(projected, 2),
            "projected_on_ci": [
                round(projected - 1.96 * se_proj, 1),
                round(projected + 1.96 * se_proj, 1),
            ],
        }
    return out


def main() -> int:
    out = compute()
    (RESULTS / "representativeness.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n"
    )
    for name, r in out.items():
        print(f"  {name}")
        print(
            f"    subset off {r['subset_off']:.2f}% ({r['n_subset']})   "
            f"rest off {r['rest_off']:.2f}% ({r['n_rest']})   "
            f"easier by {r['subset_easier_by']:+.2f} points, p = {r['representativeness_p']}"
        )
        print(
            f"    paired gain {r['paired_gain']:+.2f} points {r['paired_gain_ci']}   "
            f"McNemar b={r['mcnemar_b']} c={r['mcnemar_c']} p = {r['mcnemar_p']}"
        )
        print(
            f"    full set off {r['full_off']:.2f}% {r['full_off_ci']}   "
            f"projected on {r['projected_on']:.2f}% {r['projected_on_ci']}"
        )
    print("  wrote results/representativeness.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
