# taichu-eval-reproduction

Re-measuring two model-card numbers of [ZDTaichu5.0-9B](https://huggingface.co/TaichuAI/ZDTaichu5.0-9B)
(Zidong Taichu 5.0, a Qwen3.5 hybrid GatedDeltaNet decoder with a C-RADIOv4-H vision tower) on one RTX 4090:
**CV-Bench** (card: 86.82) and **MathVista testmini** (card: 84.50).

On the card's protocol — thinking mode on, answers extracted by an LLM — the two come out
differently. **CV-Bench reproduces, and slightly high:** every way of reading the thinking-on
subsample puts the full set at 88.45% to 89.33%, which is 1.63 to 2.51 points *above* the card's
86.82, with 86.82 inside every interval. **MathVista is not excluded, and does not reach the card
under any reading:** the same four estimates span 77.20% to 82.00% against 84.50 — short by 2.50 to
7.30 points — and 84.50 lies inside all four intervals. One hundred items cannot settle that, and
this page does not claim they do. Off the card's protocol both numbers are far lower, and that part
of the gap is the protocol rather than the weights.


> **On machine names.** Machines are referred to neutrally throughout. The
> pre-registration was sealed before that decision, so removing the name from it
> is a **redaction** and is recorded as one in
> [`prereg/REDACTION.md`](prereg/REDACTION.md), with the hash it was sealed
> under, the hash now, and the byte count showing the change is a same-length
> substitution of one token and nothing else.

## Numbers

| benchmark | setting | items | accuracy | 95% CI | card |
|---|---|---|---|---|---|
| CV-Bench, Cambrian-style overall = (mean(ADE20K, COCO) + Omni3D) / 2 | thinking off, 32-token budget | 2,638 (all) | 82.42% | [81.0, 83.9] (item-level) | 86.82 |
| CV-Bench, same items, paired | thinking off / **thinking on**, 3,072-token budget | 150 (fixed random subset) | 83.33% / **89.33%** | [83.4, 93.3] for thinking on | 86.82 |
| MathVista testmini | thinking off, 512 tokens, LLM-extracted | 1,000 (all) | 66.70% | [63.7, 69.6] | 84.50 |
| MathVista testmini | thinking off, 2,048 tokens, LLM-extracted | 1,000 (all) | 73.20% | [70.4, 75.9] | 84.50 |
| MathVista testmini, same items, paired | thinking off 2,048 / **thinking on** 3,072, LLM-extracted | 100 (fixed random subset) | 78.0% / **82.0%** | [73.3, 88.3] for thinking on | 84.50 |

The two subsample rows are unbiased estimates of the full-set numbers, but they are the widest
ones on this page and they ignore what thinking off already measured on every item. The section
below uses that and narrows them.

CV-Bench by task (thinking off, all items): ADE20K Count 54.97%, ADE20K Relation 90.38%, COCO Count 75.56%,
COCO Relation 95.82%, Omni3D Depth 92.67%, Omni3D Distance 81.17%. On the 150-item subset, paired, thinking mode's +6.00-point gain comes from Omni3D Distance
(+4 of 41) and ADE20K Count (+3 of 18, 50.00% to 66.67%). It moves COCO Count by **nothing**:
91.30% with thinking off and 91.30% with it on, one item fixed and one broken. An earlier version
of this line read "counting is where thinking pays", which the per-task decomposition does not
support.

MathVista with the rule-based extractor instead of the LLM one: 54.30% / 60.40% / 52.00% for the three rows
above. The model answers with a derivation and the answer is often not the last number, so a regex reads it
wrong; the official protocol uses GPT-4 to extract the answer, and a local Qwen2.5-7B-Instruct with the same
few-shot prompt shape closes most of that gap. Everything here is greedy decoding.

## Reading a subsample against a card number

Thinking mode was affordable on 150 CV-Bench items and 100 MathVista items, drawn uniformly at
random (`rng.sample`, seed 0). Two facts about that pull in opposite directions and are easy to
run together:

- A uniform subsample mean is an **unbiased** estimate of the full-set accuracy. "82.0% on 100
  items, and 84.50 is inside its interval" is a valid statement. An earlier version of this page
  said the subsample figure could not be read as a full-set number; that was wrong, and this
  section replaces it.
- This particular draw is **easy**. Under thinking off — the protocol both arms share, and the one
  measured on every item of both full sets — the MathVista 100 score 5.33 points above the other
  900, and the CV-Bench 150 score 0.94 points above the other 2,488:

| benchmark | subsample, thinking off | the rest, thinking off | subsample easier by | p |
|---|---|---|---|---|
| CV-Bench | 83.33% (150) | 82.40% (2,488) | +0.94 points | 0.77 |
| MathVista | 78.00% (100) | 72.67% (900) | +5.33 points | 0.25 |

Neither gap is significant — a random draw of 100 does that — so neither is evidence of a broken
sample. But the thinking-off arm is a **census** of both full sets — 82.45% over all 2,638 CV-Bench
items and 73.20% over all 1,000 MathVista items, per item in both cases — so it is an auxiliary
variable known for every item, and using it is free precision. Four standard estimates of the full-set
thinking-on accuracy, the subsample correlated with its auxiliary at rho = 0.48 (CV-Bench) and
0.51 (MathVista), with the finite-population correction applied:

| estimator | CV-Bench (card 86.82) | MathVista (card 84.50) |
|---|---|---|
| direct — the subsample mean | 89.33% ± 2.46, [84.5, 94.1] | 82.00% ± 3.66, [74.8, 89.2] |
| difference — full-set off plus the paired gain | 88.45% ± 2.79, [83.0, 93.9] | 77.20% ± 3.79, [69.8, 84.6] |
| **regression — least-squares coefficient** | **88.98% ± 2.15, [84.8, 93.2]** | **79.75% ± 3.16, [73.6, 85.9]** |
| post-stratified — by what thinking off got right | 88.98% ± 2.26, [84.5, 93.4] | 79.75% ± 3.60, [72.7, 86.8] |

Four rows, three estimators. With a binary auxiliary the regression and post-stratified
estimators are algebraically the same thing — identical point estimates and identical bootstrap
intervals, differing only in which analytic variance formula is used — and the table keeps both
rows because their standard errors are derived differently, not because they are separate
readings.

Read the regression row if you read one: it has the smallest standard error, and it is what the
auxiliary variable is for. The difference estimator is the same thing with the coefficient forced
to 1, which is right only when the arms are strongly correlated; at rho ~ 0.5 the least-squares
coefficient is 0.40 and 0.47, so forcing 1 over-corrects, and it is the least precise row in both
benchmarks. It is also the furthest from the card on MathVista — but the *closest* on CV-Bench
(1.63 points against 2.16 and 2.51), so over-correction moves it away from the card only where the
card sits above the estimates. The least-squares coefficient is 0.40 and 0.47. Percentile bootstrap
intervals over 4,000 resamples agree with all four analytic ones to within 1.1 points.

The conclusion is the same in every row, which is the point of printing all four: CV-Bench above
the card and MathVista below it, with the card inside every interval. What a larger thinking-on
sample would buy is precision, not a different sign.

`scripts/representativeness.py` derives every figure above, `results/representativeness.json`
holds them, and `scripts/check_readme_numbers.py` fails if the page and that file disagree.

## What is and is not established

- **CV-Bench: reproduced.** All four estimates of the full-set thinking-on accuracy — 88.45% to
  89.33% — are above the card's 86.82, by 1.63 to 2.51 points, and 86.82 is inside all four
  intervals. The subsample they rest on is representative (+0.94 points, p = 0.77) and the paired
  thinking-mode gain that carries them is +6.00 points (McNemar exact p = 0.064). The full-set
  thinking-off number, 82.42%, is a different protocol and should not be read against the card.
- **MathVista: not excluded, not reached — and that verdict turns on eleven items.** All four
  estimates — 77.20% to 82.00% — are below 84.50, by 2.50 to 7.30 points, and 84.50 is inside all
  four intervals. But **11 of the 100 thinking-on items never closed their reasoning block**, all
  of them at the 3,072-token cap, so there was no answer to extract; the LLM extractor read the
  tail of an unfinished trace and credited 5 of the 11 as correct. Score those 11 as no-answer and
  the estimates fall to 72.20–77.00% and **three of the four intervals stop containing 84.50**
  (`results/unclosed_sensitivity.json`). Neither scoring is obviously right — a truncated trace is
  not a wrong answer, and it is not a right one either — so the honest statement is that this
  benchmark's verdict is not robust to how eleven truncated generations are counted, and a larger
  token budget, not a larger sample, is what would settle it. Two things keep this
  weak in both directions: the paired gain is +4.00 points but is not distinguishable from zero on
  100 items (McNemar exact p = 0.45), and the draw is 5.33 points easy, which is why the
  unadjusted 82.00% sits closest to the card and the adjusted estimates sit further from it. The
  subsample is small because thinking-on generation averages 1,398 tokens per item
  (14 of 100 hit the 3,072 cap, and 11 of those never closed the block) and this is one shared GPU.
- Not established: the card's exact prompts, budgets and extractor. The comparison is "does the
  number survive an independent run of the standard protocol", not "is it bit-exact". A larger
  thinking-on MathVista sample would narrow the interval; it is the one thing 100 items cannot do.

## Running it

```bash
pip install timm                                  # the C-RADIO tower needs it
python scripts/taichu_eval.py --bench cvbench   --thinking off --max_new 32
python scripts/taichu_eval.py --bench mathvista --thinking off --max_new 2048 --tag b2048
python scripts/taichu_eval.py --bench cvbench   --thinking on  --max_new 3072 --subset 150
python scripts/taichu_eval.py --bench mathvista --thinking on  --max_new 3072 --subset 100
# Qwen2.5-7B-Instruct via vLLM, over each of the three MathVista runs
python scripts/llm_extract.py results/mathvista_think_off.jsonl \
                             results/mathvista_think_off_b2048.jsonl \
                             results/mathvista_think_on_sub100.jsonl
python scripts/taichu_summary.py                  # the Numbers table
python scripts/representativeness.py              # the subsample and estimator tables
```

Three things the shipped model code needed on transformers 5.16: `_keys_to_ignore_on_load_unexpected` is a
list where transformers now expects a set (converted before loading); the configs default to
`flash_attention_2`, which is not installed here, and the vision tower supports only eager attention
(`attn_implementation={"": "sdpa", "llm_config": "sdpa", "vision_config": "eager"}`); and the chat template
turns thinking on by default, so `--thinking off` renders the prompt with `enable_thinking=False` through the
same processor path. `results/` holds every per-item record (gold, prediction, whether the model closed its
thinking block, token count, the tail of the response) so any row above can be re-derived.

## License

MIT.
