# Pre-registration — the token budget the README says would settle it

Written **2026-09-23 01:11 (Asia/Shanghai)**, BEFORE any generation for this arm has been started.

## The question, in the README's own words

The MathVista thinking-on arm reports 82.00% on a fixed 100-item subset against the card's 84.50.
Eleven of those 100 items never closed their reasoning block, all of them at the 3,072-token cap, so
there was no answer to extract; the LLM extractor read the tail of an unfinished trace and credited
5 of the 11 as correct. Scored as no-answer instead, three of the four estimator intervals stop
containing 84.50. The README says this outright, and says what would resolve it:

> "the benchmark's verdict is not robust to how eleven truncated generations are counted, and **a
> larger token budget, not a larger sample, is what would settle it**. ... (14 of 100 hit the 3,072
> cap, and 11 of those never closed the block) and **this is one shared GPU**."

The second half of that sentence is no longer true. A second node carries idle 46 GB L40s, and a 9B
model at a larger budget fits on one with room to spare.

This concerns MathVista only. The CV-Bench thinking-on arm has **1 of 150** items unclosed, so its
verdict does not turn on truncation and it is not re-run here.

## Disclosure of order

This arm is run after the published numbers are known — it exists because the published README poses
the question. What is pre-committed below is the analysis and the interpretation of **both**
outcomes, written before any token of it is generated.

## Held fixed

- **Model**: `TaichuAI/ZDTaichu5.0-9B`, the same local snapshot.
- **Items**: the same fixed 100-item MathVista testmini subset, drawn the same way (`--subset 100`,
  seed 0). No item is added or dropped.
- **Decoding**: greedy (`do_sample=False`), thinking ON, the same chat template, `--bench mathvista`.
- **Extraction and scoring**: the same `llm_extract.py` and the same scorer, run the same way.

## What changes

- **Hardware**: one idle NVIDIA L40 (46 GB) instead of a shared RTX 4090 (24 GB).
- **Environment**: `envs/vllm028` (pyarrow is absent from the env the other arms use). transformers
  5.16.1 as before; torch differs, and is recorded. This is why arm T-A below exists.
- **Token budget**: 8,192 instead of 3,072, in arm T-B.

## The two arms

| arm | budget | purpose |
|---|---|---|
| **T-A** | 3,072 | the published configuration, re-run here. The control for hardware and environment. |
| **T-B** | 8,192 | the budget the README says would settle the question. |

Both run on gpu-03, on separate idle L40s, greedy, no time cap.

## Pre-stated analysis

**Primary (T-B).** The number of items whose reasoning block does not close, and the number at the
8,192 cap. Reported first, before any accuracy.

**Accuracy, both scorings, both arms.** Extractor as-is, and truncated-as-no-answer, exactly as
`results/unclosed_sensitivity.json` already does it. The four estimators and their bootstrap
intervals are recomputed for T-B by the existing `representativeness.py`, unchanged.

**The control (T-A vs the published run), item by item.** Agreement on `ok`, on `thought` (whether
the block closed), and the distribution of `ntok`. Pre-committed:

- If T-A agrees with the published run item for item, T-B may be compared directly against the
  published numbers.
- If it does not, T-B is still reported in full, the disagreement is reported with it, and **only the
  T-A vs T-B contrast run tonight is used** for any statement about what the budget buys. The
  published numbers are not restated as if this environment had reproduced them.

**Pre-stated interpretation, committed before the result exists.**

- **If T-B closes all or nearly all of the blocks**, the ambiguity the README flags is resolved: one
  scoring rule applies, the sensitivity table collapses to a single column, and whichever side of
  84.50 the intervals fall on is the answer. That is reported whichever side it is.
- **If items still truncate at 8,192**, the count is reported and the budget is **not** raised again.
  A budget chosen after seeing a score is not a budget, and the README's existing two-column
  sensitivity treatment stands.
- **If T-B's accuracy is lower than T-A's**, that is reported with the same prominence as if it were
  higher. A larger budget is not assumed to help.

**No further amendment will be made on the basis of seeing these scores.**

## What gets committed

Both arms' per-item jsonl (`ntok`, whether the block closed, prediction, gold), the extraction
output, the recomputed estimator table, and the environment of both arms — which no earlier run in
this repository recorded.
