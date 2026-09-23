"""Every number the README states must be re-derivable from the per-item records in results/.

Two kinds of check, because a page like this fails in two different ways.

SCORES      the ten accuracies, recomputed from results/metrics-style records -- including the
            Cambrian-style CV-Bench aggregate, which is a mean of means and not the item-level
            accuracy, so it cannot be checked by eye.

TABLES      the subsample table and the estimator table are *parsed* and compared cell by cell
            against scripts/representativeness.py. They are not searched for as loose numbers,
            and that distinction is the whole reason this file was rewritten: 79.75% appears in
            two rows of the estimator table and 5.33 appears three times on the page, so a check
            that asks "is 79.75 somewhere on the page" passes with the regression row wrong. It
            did, on the first attempt at breaking it.

The prose sentences the tables support -- the two ranges, the shortfall range, the bootstrap
tolerance, "every interval contains the card" -- are parsed out of the page too and held to the
same source, since those are the sentences a reader actually takes away.

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
NAMES = {"cvbench": "CV-Bench", "mathvista": "MathVista"}
ROW_KEY = {
    "direct": "direct",
    "difference": "difference",
    "regression": "regression",
    "post-stratified": "post_stratified",
}


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
    return 100 * (two_d + sum(by["Omni3D"]) / len(by["Omni3D"])) / 2


def _rows(readme, header_fragment):
    """The body rows of the first markdown table whose header contains this fragment."""
    lines = readme.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|") and header_fragment in line:
            body = []
            for line2 in lines[i + 2 :]:
                if not line2.startswith("|"):
                    break
                body.append([c.strip() for c in line2.strip("|").split("|")])
            return body
    return []


NUM = r"(\d+\.\d+)"


def parse_subsample_table(readme):
    out = {}
    for cells in _rows(readme, "subsample easier by"):
        bench, sub, rest, easier, p = cells[:5]
        out[bench] = {
            "subset_off": float(re.search(NUM, sub).group(1)),
            "rest_off": float(re.search(NUM, rest).group(1)),
            "subset_easier_by": float(re.search(r"\+" + NUM, easier).group(1)),
            "representativeness_p": float(p),
        }
    return out


def parse_estimator_table(readme):
    out = {}
    for cells in _rows(readme, "estimator"):
        name = re.sub(r"[*`]", "", cells[0]).split("—")[0].strip()
        if name not in ROW_KEY:
            continue
        row = {}
        for bench, cell in zip(("CV-Bench", "MathVista"), cells[1:3]):
            m = re.search(
                NUM + r"% ± " + NUM + r", \[" + NUM + ", " + NUM + r"\]",
                cell.replace("*", ""),
            )
            assert m, f"unreadable cell for {name}/{bench}: {cell}"
            row[bench] = tuple(float(g) for g in m.groups())
        out[ROW_KEY[name]] = row
    return out


def main() -> int:
    cv_off, cv_on = load("cvbench_think_off.jsonl"), load(
        "cvbench_think_on_sub150.jsonl"
    )
    mv_2048 = load("mathvista_think_off_b2048_llmext.json")
    mv_512 = load("mathvista_think_off_llmext.json")
    mv_on = load("mathvista_think_on_sub100_llmext.json")
    if any(x is None for x in (cv_off, cv_on, mv_2048, mv_512, mv_on)):
        print("  a results file is missing")
        return 1

    off_by_id = {r["id"]: r["ok"] for r in cv_off}
    paired_off = [{"ok": off_by_id[r["id"]]} for r in cv_on if r["id"] in off_by_id]
    mv_by_id = {r["id"]: r["llm_ok"] for r in mv_2048}
    mv_paired = [{"llm_ok": mv_by_id[r["id"]]} for r in mv_on if r["id"] in mv_by_id]

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    failures = []

    scores = [
        ("CV-Bench thinking off, Cambrian aggregate", cambrian(cv_off), "82.42"),
        ("CV-Bench thinking on, 150-item subsample", acc(cv_on), "89.33"),
        ("CV-Bench thinking off, the same 150 items", acc(paired_off), "83.33"),
        ("MathVista off 2,048, LLM-extracted", acc(mv_2048, "llm_ok"), "73.20"),
        ("MathVista off 512, LLM-extracted", acc(mv_512, "llm_ok"), "66.70"),
        (
            "MathVista on, 100-item subsample, LLM-extracted",
            acc(mv_on, "llm_ok"),
            "82.0",
        ),
        ("MathVista off 2,048, the same 100 items", acc(mv_paired, "llm_ok"), "78.0"),
        ("MathVista off 512, rule-based", acc(mv_512), "54.30"),
        ("MathVista off 2,048, rule-based", acc(mv_2048), "60.40"),
        ("MathVista on subsample, rule-based", acc(mv_on), "52.00"),
    ]
    for label, got, written in scores:
        places = len(written.split(".")[1])
        got_r = f"{round(got, places):.{places}f}"
        on_page = (
            re.search(r"(?<![\d.])" + re.escape(written) + r"(?![\d])", readme)
            is not None
        )
        ok = got_r == written and on_page
        print(
            f"  {label:<48} page {written:>6}  results {got_r:>6}  "
            f"{'ok' if ok else ('MISMATCH' if got_r != written else 'NOT ON PAGE')}"
        )
        if not ok:
            failures.append(label)

    for label, n, quoted in [
        ("CV-Bench items", len(cv_off), 2638),
        ("MathVista items", len(mv_2048), 1000),
        ("CV-Bench subsample", len(cv_on), 150),
        ("MathVista subsample", len(mv_on), 100),
    ]:
        on_page = any(
            re.search(r"(?<![\d.])(?<!\d,)" + re.escape(f) + r"(?!\d)(?!,\d)", readme)
            for f in {str(quoted), f"{quoted:,}"}
        )
        ok = n == quoted and on_page
        print(
            f"  {label:<48} page {quoted:>6}  results {n:>6}  "
            f"{'ok' if ok else ('MISMATCH' if n != quoted else 'NOT ON PAGE')}"
        )
        if not ok:
            failures.append(label)

    # Figures the page states that no earlier version of this file touched. A fresh-eyes audit
    # on a different model passed a README with eight of them falsified -- McNemar p 0.064 as
    # 0.640, the paired gain +6.00 as +60.0 -- and this script printed "every number on the page
    # is re-derived from results/" and exited 0.
    by_task = {}
    for r in cv_off:
        by_task.setdefault((r["source"], r["task"]), []).append(r["ok"])
    for (src, task), v in by_task.items():
        scores.append(
            (f"CV-Bench {src} {task}, all items", 100 * sum(v) / len(v), None)
        )

    sub_off = {r["id"]: r["ok"] for r in cv_off}
    sub_task = {}
    for r in cv_on:
        k = (r["source"], r["task"])
        sub_task.setdefault(k, [0, 0, 0])
        sub_task[k][0] += sub_off[r["id"]]
        sub_task[k][1] += r["ok"]
        sub_task[k][2] += 1

    tok = [r["ntok"] for r in mv_on]
    capped = sum(1 for t in tok if t >= 3072)
    unclosed = sum(1 for r in mv_on if not r["thought"])
    extras = [
        ("MathVista thinking-on mean tokens", sum(tok) / len(tok), "1,398"),
        ("MathVista items at the cap", capped, "14"),
        ("MathVista unclosed reasoning blocks", unclosed, "11"),
        (
            "CV-Bench subsample ADE20K Count, thinking off",
            100 * sub_task[("ADE20K", "Count")][0] / sub_task[("ADE20K", "Count")][2],
            "50.00",
        ),
        (
            "CV-Bench subsample ADE20K Count, thinking on",
            100 * sub_task[("ADE20K", "Count")][1] / sub_task[("ADE20K", "Count")][2],
            "66.67",
        ),
        (
            "CV-Bench subsample COCO Count, both arms",
            100 * sub_task[("COCO", "Count")][0] / sub_task[("COCO", "Count")][2],
            "91.30",
        ),
    ]
    for label, got, written in extras:
        if written is None:
            continue
        places = len(written.split(".")[1]) if "." in written else 0
        got_r = (
            f"{round(got, places):,.{places}f}"
            if "," in written
            else f"{round(got, places):.{places}f}"
        )
        on_page = (
            re.search(r"(?<![\d.])" + re.escape(written) + r"(?![\d])", readme)
            is not None
        )
        ok = got_r == written and on_page
        print(
            f"  {label:<48} page {written:>7}  results {got_r:>7}  "
            f"{'ok' if ok else ('MISMATCH' if got_r != written else 'NOT ON PAGE')}"
        )
        if not ok:
            failures.append(label)

    # COCO Count must not be described as moving: both arms are equal on the subsample
    a, b, n = sub_task[("COCO", "Count")]
    if a == b and re.search(r"COCO Count to 91\.3", readme):
        failures.append(
            "the page says thinking moves COCO Count, and on the subsample it does not"
        )

    fresh = representativeness.compute()
    full_cv = load("metrics_fullset_cvbench.json")   # load() is relative to results/
    if full_cv is None:
        sys.exit("results/metrics_fullset_cvbench.json is missing; the page's "
                 "headline measurement has nothing behind it")
    if load("representativeness.json") != fresh:
        print(
            "  results/representativeness.json is stale -- rerun scripts/representativeness.py"
        )
        return 1

    # ---- the two tables, cell by cell ----
    # Sentence-anchored, because a bare number search is satisfied by the same digits elsewhere:
    # replaying the audit's falsifications, three of eight were caught and five walked through,
    # including McNemar p 0.064 -> 0.640 and the paired gain +6.00 -> +60.0.
    sentences = [
        (
            "CV-Bench McNemar in prose",
            f"McNemar exact p = {fresh['cvbench']['mcnemar_p']:.3f}",
        ),
        (
            "MathVista McNemar in prose",
            f"McNemar exact p = {fresh['mathvista']['mcnemar_p']:.2f}",
        ),
        (
            "CV-Bench paired gain in prose",
            f"is +{fresh['cvbench']['paired_gain']:.2f} points",
        ),
        (
            "ADE20K Count, both arms on the subsample",
            (
                f"{100 * sub_task[('ADE20K', 'Count')][0] / sub_task[('ADE20K', 'Count')][2]:.2f}%"
                f" to {100 * sub_task[('ADE20K', 'Count')][1] / sub_task[('ADE20K', 'Count')][2]:.2f}%"
            ),
        ),
        (
            "COCO Count, unchanged by thinking",
            (
                f"{100 * sub_task[('COCO', 'Count')][0] / sub_task[('COCO', 'Count')][2]:.2f}%"
                " with thinking off"
            ),
        ),
        ("tokens and the cap", f"averages {sum(tok) / len(tok):,.0f} tokens per item"),
        (
            "how many of the capped never closed",
            f"{capped} of 100 hit the 3,072 cap, and {unclosed} of those never closed the block",
        ),
    ]
    for label, phrase in sentences:
        ok = phrase in readme
        print(f"  sentence  {label:<44} {'ok' if ok else 'NOT ON PAGE'}")
        if not ok:
            failures.append(f"{label}: the page does not say {phrase!r}")

    page_sub = parse_subsample_table(readme)
    page_est = parse_estimator_table(readme)
    for key, r in fresh.items():
        bench = NAMES[key]
        got = page_sub.get(bench)
        if got is None:
            failures.append(f"{bench}: no row in the subsample table")
            continue
        for field in (
            "subset_off",
            "rest_off",
            "subset_easier_by",
            "representativeness_p",
        ):
            ok = abs(got[field] - r[field]) < 5e-3
            print(
                f"  subsample table  {bench:<10} {field:<26} page {got[field]:>7}  "
                f"results {r[field]:>7}  {'ok' if ok else 'MISMATCH'}"
            )
            if not ok:
                failures.append(
                    f"{bench}.{field}: page {got[field]}, results {r[field]}"
                )

    for est, r in sorted(fresh["cvbench"]["estimators"].items()):
        if est not in page_est:
            failures.append(f"the estimator table has no {est} row")
    for est, page_row in sorted(page_est.items()):
        for key, bench in NAMES.items():
            e = fresh[key]["estimators"][est]
            want = (e["value"], e["se"], e["ci"][0], e["ci"][1])
            ok = all(abs(a - b) < 5e-3 for a, b in zip(page_row[bench], want))
            print(
                f"  estimator table  {bench:<10} {est:<16} page {page_row[bench]}  "
                f"results {want}  {'ok' if ok else 'MISMATCH'}"
            )
            if not ok:
                failures.append(
                    f"{bench}.{est}: page {page_row[bench]}, results {want}"
                )

    # ---- the sentences the tables support ----
    def vals(pattern, where=readme):
        """Match a sentence regardless of where it happens to wrap.

        These patterns used literal spaces and literal newlines, so rewrapping
        a paragraph broke four checks at once and the guard reported the page
        as disagreeing with results/ when only the line breaks had moved. A
        guard that fails on reflow is a guard people start ignoring.
        """
        m = re.search(re.sub(r"(?:\\n|[ ])+", r"\\s+", pattern), where)
        return tuple(float(g) for g in m.groups()) if m else None

    cv_vals = [e["value"] for e in fresh["cvbench"]["estimators"].values()]
    mv_vals = [e["value"] for e in fresh["mathvista"]["estimators"].values()]
    sentences = [
        # CV-Bench is no longer estimated on this page: it is measured, and the
        # estimators are reported as errors against that measurement. So these
        # check the sentences that are actually there.
        (
            "the CV-Bench measurement",
            vals(r"budget, \*\*" + NUM + r"%\*\*"),
            (full_cv["accuracy_pct"],),
        ),
        (
            "the CV-Bench interval",
            vals(r"\*\*\[" + NUM + r", " + NUM + r"\]\*\*"),
            tuple(full_cv["ci_pct"]),
        ),
        (
            "the CV-Bench distance from the card",
            vals(r"\+" + NUM + r" points away"),
            (abs(full_cv["accuracy_pct"] - full_cv["card"]),),
        ),
        (
            "the estimators' signed errors",
            vals(r"by \+" + NUM + r" to \+" + NUM + r" points"),
            (min(e["signed_error"] for e in full_cv["estimators"].values()),
             max(e["signed_error"] for e in full_cv["estimators"].values())),
        ),
        (
            "the MathVista range",
            vals(r"estimates span " + NUM + r"% to " + NUM + "%"),
            (min(mv_vals), max(mv_vals)),
        ),
        (
            "the MathVista shortfall",
            vals(r"short by " + NUM + r" to\n" + NUM + r" points"),
            (fresh["mathvista"]["shortfall_min"], fresh["mathvista"]["shortfall_max"]),
        ),
        (
            "the correlations",
            vals(r"rho = " + NUM + r" \(CV-Bench\) and\n" + NUM + r" \(MathVista\)"),
            (fresh["cvbench"]["correlation"], fresh["mathvista"]["correlation"]),
        ),
        (
            "the least-squares coefficients",
            vals(r"least-squares coefficient is " + NUM + " and " + NUM),
            (
                fresh["cvbench"]["regression_coefficient"],
                fresh["mathvista"]["regression_coefficient"],
            ),
        ),
    ]
    for label, page_v, want in sentences:
        ok = page_v is not None and all(
            abs(a - b) < 5.1e-3 for a, b in zip(page_v, want)
        )
        print(
            f"  sentence  {label:<38} page {page_v}  results "
            f"{tuple(round(w, 3) for w in want)}  {'ok' if ok else 'MISMATCH'}"
        )
        if not ok:
            failures.append(label)

    # the bootstrap tolerance the page promises, read off the page rather than hardcoded
    m = re.search(
        r"agree with all four analytic ones to within " + NUM + " points", readme
    )
    if not m:
        failures.append("the page no longer states a bootstrap agreement tolerance")
    else:
        claimed = float(m.group(1))
        worst = max(
            abs(e["ci"][i] - e["bootstrap_ci"][i])
            for r in fresh.values()
            for e in r["estimators"].values()
            for i in (0, 1)
        )
        ok = worst <= claimed + 1e-9
        print(
            f"  sentence  {'bootstrap agreement':<38} page {claimed}  worst {worst:.1f}  "
            f"{'ok' if ok else 'MISMATCH'}"
        )
        if not ok:
            failures.append(
                f"the page promises {claimed} points, the worst gap is {worst:.1f}"
            )

    for key, above in (("cvbench", True), ("mathvista", False)):
        r = fresh[key]
        holds = all((e["value"] > r["card"]) == above for e in r["estimators"].values())
        if not holds:
            failures.append(
                f"the page says every {NAMES[key]} estimate is "
                f"{'above' if above else 'below'} the card; one is not"
            )
        if not r["every_interval_contains_card"]:
            failures.append(
                f"{NAMES[key]}: the page says every interval contains the card"
            )

    if failures:
        print("\n  the page no longer matches results/: " + "; ".join(failures))
        return 1
    print("  every number on the page is re-derived from results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
