"""Every accuracy the README states must be re-derivable from the per-item records in results/.

The README quotes a dozen figures: two model-card numbers it is testing, and ten of its own. The
ten are recomputed here from the JSONL and JSON files this repository commits -- including the
Cambrian-style CV-Bench aggregate, which is a mean of means and not the item-level accuracy, so it
cannot be checked by eye. A figure that no longer matches, to the precision the README writes it at,
fails the build rather than quietly staying on the page.

    python scripts/check_readme_numbers.py
"""
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load(name):
    p = RESULTS / name
    if not p.exists():
        return None
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return json.load(open(p))


def acc(rows, key="ok"):
    return 100 * sum(r[key] for r in rows) / len(rows)


def cambrian(rows):
    """CV-Bench overall as the Cambrian-1 tables report it: (mean(ADE20K, COCO) + Omni3D) / 2."""
    by = {}
    for r in rows:
        by.setdefault(r["source"], []).append(r["ok"])
    two_d = (sum(by["ADE20K"]) / len(by["ADE20K"]) + sum(by["COCO"]) / len(by["COCO"])) / 2
    three_d = sum(by["Omni3D"]) / len(by["Omni3D"])
    return 100 * (two_d + three_d) / 2


def main() -> int:
    cv_off, cv_on = load("cvbench_think_off.jsonl"), load("cvbench_think_on_sub150.jsonl")
    mv_llm_2048 = load("mathvista_think_off_b2048_llmext.json")
    mv_llm_512 = load("mathvista_think_off_llmext.json")
    mv_llm_on = load("mathvista_think_on_sub100_llmext.json")
    if any(x is None for x in (cv_off, cv_on, mv_llm_2048, mv_llm_512, mv_llm_on)):
        print("  a results file is missing")
        return 1

    off_by_id = {r["id"]: r["ok"] for r in cv_off}
    paired_off = [{"ok": off_by_id[r["id"]]} for r in cv_on if r["id"] in off_by_id]

    mv2048_by_id = {r["id"]: r["llm_ok"] for r in mv_llm_2048}
    mv_paired_off = [{"llm_ok": mv2048_by_id[r["id"]]} for r in mv_llm_on if r["id"] in mv2048_by_id]

    # (label, value recomputed here, the figure exactly as the README writes it)
    checks = [
        ("CV-Bench thinking off, Cambrian aggregate", cambrian(cv_off), "82.42"),
        ("CV-Bench thinking on, 150-item subset", acc(cv_on), "89.33"),
        ("CV-Bench thinking off, the same 150 items", acc(paired_off), "83.33"),
        ("MathVista off 2,048, LLM-extracted", acc(mv_llm_2048, "llm_ok"), "73.20"),
        ("MathVista off 512, LLM-extracted", acc(mv_llm_512, "llm_ok"), "66.70"),
        ("MathVista on, 100-item subset, LLM-extracted", acc(mv_llm_on, "llm_ok"), "82.0"),
        ("MathVista off 2,048, the same 100 items", acc(mv_paired_off, "llm_ok"), "78.0"),
        ("MathVista off 512, rule-based", acc(mv_llm_512), "54.30"),
        ("MathVista off 2,048, rule-based", acc(mv_llm_2048), "60.40"),
        ("MathVista on subset, rule-based", acc(mv_llm_on), "52.00"),
    ]
    readme = (ROOT / "README.md").read_text()
    failures = []
    for label, got, written in checks:
        places = len(written.split(".")[1])
        got_r = f"{round(got, places):.{places}f}"
        in_page = re.search(r"(?<![\d.])" + re.escape(written) + r"(?![\d])", readme) is not None
        ok = got_r == written and in_page
        state = "ok" if ok else ("MISMATCH" if got_r != written else "NOT ON PAGE")
        print(f"  {label:<46} page {written:>6}  results {got_r:>6}  {state}")
        if not ok:
            failures.append(label)
    # the item counts the page states
    for label, n, quoted in [("CV-Bench items", len(cv_off), 2638), ("MathVista items", len(mv_llm_2048), 1000),
                             ("CV-Bench subset", len(cv_on), 150), ("MathVista subset", len(mv_llm_on), 100)]:
        ok = n == quoted
        print(f"  {label:<46} page {quoted:>6}  results {n:>6}  {'ok' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(label)
    if failures:
        print("\n  the page no longer matches results/: " + "; ".join(failures))
        return 1
    print("  every number on the page is re-derived from results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
