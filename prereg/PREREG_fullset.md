# Pre-registration — the full sets, which retire the estimators

Written **2026-09-23 10:15 (Asia/Shanghai)**, BEFORE any generation for these arms.

## Why these arms exist

This repository reports **four estimators** of the full-set thinking-on accuracy
— direct, difference, regression, post-stratified — because thinking mode was
affordable only on a random subsample. The estimators are legitimate and they
disagree by several points, which is why all four are reported.

They exist because of a compute limitation, and nothing else. **That limitation
is gone**: idle accelerators are available in quantity, and a budget arm
(`PREREG_budget.md`) has already established that an 8192-token budget leaves
2 of 100 items truncated rather than 15.

So the honest thing is to measure the quantity the estimators estimate.

**Disclosure of order.** These arms are being run after the subsample results,
the budget arms and the estimator table are all known. The reason is hardware
availability. Nothing in the analysis below is chosen with knowledge of a
full-set score, because no full-set score exists yet — and the comparison
between the estimators and the measured value is pre-stated here precisely so
that it cannot be selected afterwards.

## Held fixed

- **Model**: the same local snapshot, bf16, greedy (`do_sample=False`), thinking
  ON, the same chat template.
- **Data**: the same local benchmark copies the existing runs used.
- **Extraction and scoring**: the same LLM extractor and the same scorer, run the
  same way as the subsample arms.
- **Budget**: 8192 tokens, as in arm T-B.

## What changes

- **Every item**, not a subsample: MathVista testmini **1000**, CV-Bench **2638**.
- **Hardware**: idle accelerators, sharded.

## The two arms, and one rule about how they are split

| arm | benchmark | items | accelerator |
|---|---|---|---|
| **F-MV** | MathVista testmini | 1000 | one model of accelerator, sharded |
| **F-CV** | CV-Bench | 2638 | a different model, sharded |

**Neither arm is split across two accelerator models.** Decoding here is one
item at a time, so sharding is a throughput knob and not a batch-composition
change — but the model of card is not a throughput knob, and this repository's
companion measurements found the same weights and seed scoring differently on
different cards. Each arm therefore runs entirely on one model of accelerator,
and which one is recorded.

Shards own `items[i::n]` of the fixed list, sliced before the already-done
filter, so a restarted shard keeps its items. Shard files are merged with a
check that any id appearing twice appears identically.

## Pre-stated analysis

**Primary.** The full-set thinking-on accuracy for each benchmark, with an exact
(Clopper–Pearson) 95% interval, against the model card's number. This is a
direct measurement and **it supersedes the four estimators**, which were
estimates of exactly this quantity.

**The estimators, judged.** For each benchmark, report every estimator's point
value and interval next to the measured value, and the signed error of each.
Pre-stated, whichever way it comes out:

- if an estimator's interval contains the measured value, say so;
- if one does not, say which and by how much;
- **do not** retrospectively prefer whichever estimator turns out closest. The
  ranking is reported as a result about estimators on this data, not as a
  recommendation.

**Truncation.** The count of items whose reasoning block does not close at 8192,
on the full set. If any remain, the accuracy is reported under both scorings —
extractor as-is, and truncated-as-no-answer — exactly as
`results/unclosed_sensitivity.json` already does.

**A determinism control, for free.** The 100 subsample items are inside F-MV at
the same budget as arm T-B. Their per-item agreement with T-B is reported:
tokens identical, outcome identical, text identical. This is the control the
budget arms lacked.

**Interpretation, committed in advance.**

- If the measured full-set value is **below** the card's number, that is stated
  plainly, with the interval, and the reproduction's verdict rests on a
  measurement rather than on an estimator.
- If it is **above**, the same.
- If it **lands inside the spread of the four estimators**, that is not a
  vindication of the estimator approach; it is one draw.
- If an arm cannot complete, the completed count is reported, labelled partial,
  and the estimator treatment stands for that benchmark with its existing
  wording.

**No further amendment will be made on the basis of seeing these scores.**

## What gets committed

Per-item results for every shard, the merged file, the extraction output, the
recomputed estimator comparison, and the accelerator model and environment of
each arm.
