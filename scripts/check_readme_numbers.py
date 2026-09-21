"""Every number the README states must be re-derivable from the per-item records in results/.

The README quotes two model-card numbers it is testing and thirty of its own. The thirty are
recomputed here from the JSONL and JSON files this repository commits -- including the Cambrian-style
CV-Bench aggregate, which is a mean of means and not the item-level accuracy, and the whole
subset-representativeness argument, neither of which can be checked by eye. A figure that no longer
matches, to the precision the README writes it at, fails the build rather than quietly staying on
the page.

The representativeness figures are checked twice over: against results/representativeness.json as
committed, and against a fresh recomputation from the raw records, so neither the page nor the
generated file can drift on its own.

What this cannot catch: it asks whether a figure appears *somewhere* on the page, not whether it
appears in the right row, so a number swapped between two rows would pass. It exists to stop a
figure from surviving a change to results/, which is the way these pages actually go stale.

    python scripts/check_readme_numbers.py
"""

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import representativeness  # imported after the sys.path line above

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load(name):
    p = RESULTS / name
    if not p.exists():
        return None
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return json.loads(p.read_text())


def acc(rows, key="ok"):
    return 100 * sum(r[key] for r in rows) / len(rows)


def cambrian(rows):
    """CV-Bench overall as the Cambrian-1 tables report it: (mean(ADE20K, COCO) + Omni3D) / 2."""
    by = {}
    for r in rows:
        by.setdefault(r["source"], []).append(r["ok"])
    two_d = (
        sum(by["ADE20K"]) / len(by["ADE20K"]) + sum(by["COCO"]) / len(by["COCO"])
    ) / 2
    three_d = sum(by["Omni3D"]) / len(by["Omni3D"])
    return 100 * (two_d + three_d) / 2


SIGNS = "+-\u2212"  # the page writes a Unicode minus in intervals


def figure_matches(got, written, readme):
    """Does `got` round to the figure the page writes, and is that figure actually on the page?"""
    body = written.lstrip(SIGNS)
    places = len(body.split(".")[1]) if "." in body else 0
    sign = -1 if written[0] in "-\u2212" else 1
    got_r = f"{sign * round(abs(got), places) if got else 0.0:.{places}f}"
    expected = f"{sign * float(body):.{places}f}"
    on_page = (
        re.search(
            r"(?<![\d.])" + re.escape(body.replace(",", ",")) + r"(?![\d])", readme
        )
        is not None
    )
    return got_r == expected, got_r, on_page


def main() -> int:
    cv_off, cv_on = (
        load("cvbench_think_off.jsonl"),
        load("cvbench_think_on_sub150.jsonl"),
    )
    mv_llm_2048 = load("mathvista_think_off_b2048_llmext.json")
    mv_llm_512 = load("mathvista_think_off_llmext.json")
    mv_llm_on = load("mathvista_think_on_sub100_llmext.json")
    if any(x is None for x in (cv_off, cv_on, mv_llm_2048, mv_llm_512, mv_llm_on)):
        print("  a results file is missing")
        return 1

    off_by_id = {r["id"]: r["ok"] for r in cv_off}
    paired_off = [{"ok": off_by_id[r["id"]]} for r in cv_on if r["id"] in off_by_id]

    mv2048_by_id = {r["id"]: r["llm_ok"] for r in mv_llm_2048}
    mv_paired_off = [
        {"llm_ok": mv2048_by_id[r["id"]]} for r in mv_llm_on if r["id"] in mv2048_by_id
    ]

    # (label, value recomputed here, the figure exactly as the README writes it)
    checks = [
        ("CV-Bench thinking off, Cambrian aggregate", cambrian(cv_off), "82.42"),
        ("CV-Bench thinking on, 150-item subset", acc(cv_on), "89.33"),
        ("CV-Bench thinking off, the same 150 items", acc(paired_off), "83.33"),
        ("MathVista off 2,048, LLM-extracted", acc(mv_llm_2048, "llm_ok"), "73.20"),
        ("MathVista off 512, LLM-extracted", acc(mv_llm_512, "llm_ok"), "66.70"),
        (
            "MathVista on, 100-item subset, LLM-extracted",
            acc(mv_llm_on, "llm_ok"),
            "82.0",
        ),
        (
            "MathVista off 2,048, the same 100 items",
            acc(mv_paired_off, "llm_ok"),
            "78.0",
        ),
        ("MathVista off 512, rule-based", acc(mv_llm_512), "54.30"),
        ("MathVista off 2,048, rule-based", acc(mv_llm_2048), "60.40"),
        ("MathVista on subset, rule-based", acc(mv_llm_on), "52.00"),
    ]
    readme = (ROOT / "README.md").read_text()
    failures = []

    # the representativeness argument: subset-vs-rest, the paired gain, and what it projects to
    fresh = representativeness.compute()
    committed = load("representativeness.json")
    if committed is None:
        print("  results/representativeness.json is missing")
        return 1
    if committed != fresh:
        print(
            "  results/representativeness.json is stale -- rerun scripts/representativeness.py"
        )
        return 1
    cv, mv = fresh["cvbench"], fresh["mathvista"]
    checks += [
        ("CV-Bench 150 items under thinking off", cv["subset_off"], "83.33"),
        ("CV-Bench the other 2,488 under thinking off", cv["rest_off"], "82.40"),
        ("CV-Bench subset easier by", cv["subset_easier_by"], "+0.94"),
        ("CV-Bench subset-vs-rest p", cv["representativeness_p"], "0.77"),
        ("CV-Bench paired gain", cv["paired_gain"], "+6.00"),
        ("CV-Bench paired gain CI low", cv["paired_gain_ci"][0], "0.37"),
        ("CV-Bench paired gain CI high", cv["paired_gain_ci"][1], "11.63"),
        ("CV-Bench McNemar p", cv["mcnemar_p"], "0.064"),
        ("CV-Bench full set, per-item, thinking off", cv["full_off"], "82.45"),
        ("CV-Bench projected thinking on", cv["projected_on"], "88.45"),
        ("CV-Bench projected CI low", cv["projected_on_ci"][0], "82.6"),
        ("CV-Bench projected CI high", cv["projected_on_ci"][1], "94.3"),
        ("MathVista 100 items under thinking off", mv["subset_off"], "78.00"),
        ("MathVista the other 900 under thinking off", mv["rest_off"], "72.67"),
        ("MathVista subset easier by", mv["subset_easier_by"], "+5.33"),
        ("MathVista subset-vs-rest p", mv["representativeness_p"], "0.25"),
        ("MathVista paired gain", mv["paired_gain"], "+4.00"),
        ("MathVista paired gain CI low", mv["paired_gain_ci"][0], "\u22123.84"),
        ("MathVista paired gain CI high", mv["paired_gain_ci"][1], "11.84"),
        ("MathVista McNemar p", mv["mcnemar_p"], "0.45"),
        ("MathVista projected thinking on", mv["projected_on"], "77.20"),
        ("MathVista projected CI low", mv["projected_on_ci"][0], "68.9"),
        ("MathVista projected CI high", mv["projected_on_ci"][1], "85.5"),
        # figures the prose derives from the tables rather than restating
        ("MathVista card minus the estimate", 84.50 - mv["projected_on"], "7.30"),
        (
            "CV-Bench per-item minus Cambrian accounting",
            cv["full_off"] - cambrian(cv_off),
            "0.03",
        ),
    ]

    for label, got, written in checks:
        ok_value, got_r, on_page = figure_matches(got, written, readme)
        ok = ok_value and on_page
        state = "ok" if ok else ("MISMATCH" if not ok_value else "NOT ON PAGE")
        print(f"  {label:<46} page {written:>7}  results {got_r:>7}  {state}")
        if not ok:
            failures.append(label)
    # the item counts the page states
    for label, n, quoted in [
        ("CV-Bench items", len(cv_off), 2638),
        ("MathVista items", len(mv_llm_2048), 1000),
        ("CV-Bench subset", len(cv_on), 150),
        ("MathVista subset", len(mv_llm_on), 100),
        ("CV-Bench items outside the subset", cv["n_rest"], 2488),
        ("MathVista items outside the subset", mv["n_rest"], 900),
        ("CV-Bench McNemar b (on right, off wrong)", cv["mcnemar_b"], 14),
        ("CV-Bench McNemar c (off right, on wrong)", cv["mcnemar_c"], 5),
        ("MathVista McNemar b (on right, off wrong)", mv["mcnemar_b"], 10),
        ("MathVista McNemar c (off right, on wrong)", mv["mcnemar_c"], 6),
    ]:
        # the page writes thousands with a comma, so accept either spelling; a comma that is
        # not followed by a digit is punctuation ("b=10, c=6"), not a thousands separator
        on_page = any(
            re.search(
                r"(?<![\d.])(?<!\d,)" + re.escape(form) + r"(?!\d)(?!,\d)", readme
            )
            for form in {str(quoted), f"{quoted:,}"}
        )
        ok = n == quoted and on_page
        state = "ok" if ok else ("MISMATCH" if n != quoted else "NOT ON PAGE")
        print(f"  {label:<46} page {quoted:>7}  results {n:>7}  {state}")
        if not ok:
            failures.append(label)
    if failures:
        print("\n  the page no longer matches results/: " + "; ".join(failures))
        return 1
    print("  every number on the page is re-derived from results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
