# The instrument: how much the answer extractor moves on a file that never changes

The MathVista numbers in this repository are produced by an **LLM** answer
extractor, not a regular expression, so the thing that scores the model is
itself a model. Before any arm-to-arm difference is quoted, that instrument's
own spread has to be on disk. It is measured here, ten times, on one unchanged
file.

## Setup

`results/mathvista_think_on_sub100.jsonl` — 100 generations, unchanged — scored
ten times by `scripts/llm_extract.py`, each in a fresh process. Five runs feed
the items in the recorded order; five feed the **same items permuted**, which
changes which prompts share a batch without changing the item set. Extractor:
Qwen2.5-7B-Instruct, bf16, `temperature=0.0`, `max_tokens=32`. Full environment
in `results/env.txt`.

    groundwork noise --n 10 --command \
      'python scripts/extractor_noise_probe.py results/mathvista_think_on_sub100.jsonl'

## Result

| | |
|---|---|
| accuracy, each of 10 runs | **82.00%** |
| spread across 10 runs | **0.00 points** |
| items whose extracted *text* differed across runs | **1 of 100** |
| items whose *scored outcome* differed | **0 of 100** |

Machine-readable: `results/extractor_noise_asis.json`,
`results/extractor_noise_shuffled.json`, `results/extractor_noise_itemlevel.json`.

## What that does and does not say

The aggregate is quantised: with 100 items, one flipped item is 1.00 point. So
"spread 0.00" here means **no item flipped**, not "the spread was smaller than
the resolution" — which is why the item-level column is reported next to it. An
aggregate of zero can also be two items flipping in opposite directions, and
that is ruled out only by looking at the items.

**The extractor is not deterministic.** One item of 100 produced two different
32-token outputs across the ten runs, eight times one and twice the other. It
is item 641, where the model's response contains no numerical answer at all;
one extraction says so and the other asks for more information. Both score
wrong, so the instrument's variability lands in a place where it cannot move
the number.

That is a property of **this file**, not a guarantee. An item where the
extractor sometimes recovers a number and sometimes does not would flip, and
would be worth a full point. The honest statement is: on these 100 items the
instrument's noise floor is 0.00 points, measured, and the mechanism is that
its one unstable item is unstable in a region the scoring cannot see.

## For comparison, the same measurement elsewhere

The official **rule-based** IFEval scorer, run ten times on one unchanged file
of 541 prompts, [spans 0.37 points and disagrees with itself on 2
prompts](https://github.com/GuoCheng24/ifeval-reproduction). Three runs had
said it was stable.

So the LLM extractor used here is, on this evidence, *more* stable in its
scored outcome than a published deterministic-looking scorer. That is not an
argument for LLM scoring. It is an argument for measuring the instrument you
have rather than assuming which kind is the reliable one.

## Why ten and not three

Three runs of the IFEval scorer said it was stable; the fourth did not. That is
the reason the default here is ten.
