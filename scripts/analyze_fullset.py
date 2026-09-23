#!/usr/bin/env python3
"""The pre-stated analysis of the full-set arms, and only that.

`prereg/PREREG_fullset.md` fixes what is computed here before any of it was
generated: the measured accuracy with an exact interval against the card, every
estimator judged against that measurement whichever way it comes out, the
truncation count, and - where the settings actually match - a determinism
control. Nothing is chosen after seeing a number.

    python scripts/analyze_fullset.py cvbench
    python scripts/analyze_fullset.py mathvista
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    from groundwork.stats import clopper_pearson, mcnemar_exact
except ImportError:                                  # a sibling checkout, not installed
    _sib = os.path.join(os.path.dirname(ROOT), "groundwork")
    if os.path.isdir(os.path.join(_sib, "groundwork")):
        sys.path.insert(0, _sib)
    from groundwork.stats import clopper_pearson, mcnemar_exact   # noqa: E402

# `pip install groundwork-research` is the supported route; the fallback above
# exists so this runs from a checkout sitting next to it. A hard-coded absolute
# path would be both unreproducible and a private path in a public file.

# `score` is the field the card's protocol scores on. CV-Bench is
# multiple-choice and scored by rule; MathVista's answers are pulled out by an
# LLM extractor, so its scored file is the extractor's output and its field is
# `llm_ok`. Reading `ok` there would silently report the rule-based number -
# 60.00% against the extractor's - under a heading that says otherwise.
BENCH = {
    "cvbench": {"full": "results/cvbench_think_on_FCV.jsonl",
                "subset": "results/cvbench_think_on_sub150.jsonl",
                "subset_budget": 3072, "full_budget": 8192,
                "key": "cvbench", "score": "ok"},
    "mathvista": {"full": "results/mathvista_think_on_FMV_llmext.json",
                  "subset": "results/mathvista_think_on_sub100_TB8192_llmext.json",
                  "subset_budget": 8192, "full_budget": 8192,
                  "key": "mathvista", "score": "llm_ok"},
}


def rows(path):
    """`.jsonl` one object per line, `.json` a single array - the extractor
    writes the second."""
    full = os.path.join(ROOT, path)
    with open(full, encoding="utf-8") as fh:
        if path.endswith(".jsonl"):
            return [json.loads(ln) for ln in fh if ln.strip()]
        return json.load(fh)


def main(argv):
    if not argv or argv[0] not in BENCH:
        sys.exit(f"usage: analyze_fullset.py {{{'|'.join(BENCH)}}}")
    name = argv[0]
    cfg = BENCH[name]
    full = rows(cfg["full"])
    with open(os.path.join(ROOT, "results", "representativeness.json"), encoding="utf-8") as fh:
        rep = json.load(fh)[cfg["key"]]

    n = len(full)
    sc = cfg["score"]
    k = sum(1 for r in full if r[sc])
    lo, hi = clopper_pearson(k, n)
    card = rep["card"]
    out = {"benchmark": name, "scored_on": sc, "n": n, "correct": k, "accuracy_pct": 100 * k / n,
           "ci_pct": [100 * lo, 100 * hi], "card": card,
           "interval_contains_card": 100 * lo <= card <= 100 * hi,
           "budget": cfg["full_budget"]}

    print(f"# {name}: the measurement\n")
    print(f"  {k}/{n} = {out['accuracy_pct']:.2f}%   exact 95% CI "
          f"[{out['ci_pct'][0]:.2f}, {out['ci_pct'][1]:.2f}]")
    print(f"  card {card}: the interval "
          f"{'contains' if out['interval_contains_card'] else 'EXCLUDES'} it")
    print(f"  signed difference from the card: {out['accuracy_pct'] - card:+.2f} points\n")

    # --- truncation, pre-stated ---
    unclosed = [r for r in full if not r.get("thought", True)]
    at_cap = [r for r in full if r["ntok"] >= cfg["full_budget"]]
    out["unclosed"] = len(unclosed)
    out["at_cap"] = len(at_cap)
    print(f"# truncation at {cfg['full_budget']}\n")
    print(f"  reasoning block never closed: {len(unclosed)} of {n}")
    print(f"  generation hit the cap:       {len(at_cap)} of {n}")
    if unclosed:
        k2 = sum(1 for r in full if r[sc] and r.get("thought", True))
        lo2, hi2 = clopper_pearson(k2, n)
        out["accuracy_unclosed_as_wrong_pct"] = 100 * k2 / n
        out["ci_unclosed_as_wrong_pct"] = [100 * lo2, 100 * hi2]
        print(f"  scored with unclosed as no-answer: {100 * k2 / n:.2f}% "
              f"[{100 * lo2:.2f}, {100 * hi2:.2f}]")
    else:
        print("  so there is one scoring, not two: the sensitivity that the budget\n"
              "  arms needed does not arise here.")
    print()

    # --- the estimators, judged ---
    print("# the four estimators, judged against the measurement\n")
    print(f"  {'estimator':<18}{'value':>8}{'signed err':>12}   interval          contains measured")
    est = {}
    for ename, e in sorted(rep["estimators"].items()):
        v, ci = e["value"], e["ci"]
        contains = ci[0] <= out["accuracy_pct"] <= ci[1]
        est[ename] = {"value": v, "ci": ci, "signed_error": v - out["accuracy_pct"],
                      "interval_contains_measured": contains}
        print(f"  {ename:<18}{v:8.2f}{v - out['accuracy_pct']:+12.2f}   "
              f"[{ci[0]:.1f}, {ci[1]:.1f}]".ljust(56)
              + ("yes" if contains else "NO"))
    out["estimators"] = est
    n_in = sum(1 for e in est.values() if e["interval_contains_measured"])
    errs = [e["signed_error"] for e in est.values()]
    out["estimators_containing_measured"] = n_in
    out["signed_error_range"] = [min(errs), max(errs)]
    print(f"\n  {n_in} of {len(est)} intervals contain the measured value; the point")
    print(f"  estimates are {min(errs):+.2f} to {max(errs):+.2f} points from it.")
    print("  This is reported as a result about these estimators on this data. No\n"
          "  estimator is preferred retrospectively for having come closest.\n")

    # --- the overlap, labelled for what it is ---
    sub = rows(cfg["subset"])
    same_settings = cfg["subset_budget"] == cfg["full_budget"]
    byid = {r["id"]: r for r in full}
    pairs = [(s, byid[s["id"]]) for s in sub if s["id"] in byid]
    if pairs:
        tok = sum(1 for a, b in pairs if a["ntok"] == b["ntok"])
        same = sum(1 for a, b in pairs if a[sc] == b[sc])
        # `text` is stored as a SUFFIX of the generation - text[-800:] now,
        # text[-300:] when the subsample was run - so comparing the fields
        # whole says only that the logging changed. The shared suffix is the
        # part both runs actually recorded.
        m = min(min(len(a.get("text", "")) for a, _ in pairs),
                min(len(b.get("text", "")) for _, b in pairs))
        txt = sum(1 for a, b in pairs if a.get("text", "")[-m:] == b.get("text", "")[-m:])
        bb = sum(1 for a, b in pairs if a[sc] and not b[sc])
        cc = sum(1 for a, b in pairs if b[sc] and not a[sc])
        out["overlap"] = {"n": len(pairs), "same_settings": same_settings,
                          "tokens_identical": tok, "outcome_identical": same,
                          "shared_suffix_chars": m, "suffix_identical": txt, "b": bb, "c": cc,
                          "mcnemar_p": mcnemar_exact(bb, cc)}
        title = ("# determinism control: the subsample items, same budget"
                 if same_settings else
                 f"# the {len(pairs)} overlapping items, at TWO DIFFERENT budgets")
        print(title + "\n")
        print(f"  n={len(pairs)}   tokens identical {tok}   "
              f"last {m} chars identical {txt}   outcome identical {same}")
        print(f"  discordant b={bb} c={cc}   exact McNemar p={out['overlap']['mcnemar_p']:.3f}")
        if not same_settings:
            print(f"\n  NOT the determinism control. The subsample ran at "
                  f"{cfg['subset_budget']} and the full\n  set at {cfg['full_budget']}, so "
                  "this is a budget comparison and nothing here\n  says anything about "
                  "run-to-run determinism.")
        print()

    dst = os.path.join(ROOT, "results", f"metrics_fullset_{name}.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {os.path.relpath(dst, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
