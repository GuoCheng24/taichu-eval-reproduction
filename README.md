# taichu-eval-reproduction

Re-measuring two model-card numbers of [ZDTaichu5.0-9B](https://huggingface.co/TaichuAI/ZDTaichu5.0-9B)
(Zidong Taichu 5.0, a Qwen3.5 hybrid GatedDeltaNet decoder with a C-RADIOv4-H vision tower) on one RTX 4090:
**CV-Bench** (card: 86.82) and **MathVista testmini** (card: 84.50).

On the card's protocol — thinking mode on, answers extracted by an LLM — the two numbers come out differently.
**CV-Bench reproduces:** carrying the paired thinking-mode gain onto the full set gives 88.45% [82.6, 94.3],
which contains 86.82. **MathVista does not reproduce at the point estimate:** the same construction gives
77.20% [68.9, 85.5] against a card value of 84.50, so the card is not excluded but sits 7.30 points above the
estimate, near the top of the interval. Off the card's protocol both are far lower, and that part of the gap
is the protocol rather than the weights.

## Numbers

| benchmark | setting | items | accuracy | 95% CI | card |
|---|---|---|---|---|---|
| CV-Bench, Cambrian-style overall = (mean(ADE20K, COCO) + Omni3D) / 2 | thinking off, 32-token budget | 2,638 (all) | 82.42% | [81.0, 83.9] (item-level) | 86.82 |
| CV-Bench, same items, paired | thinking off / **thinking on**, 3,072-token budget | 150 (fixed random subset) | 83.33% / **89.33%** | [83.4, 93.3] for thinking on | 86.82 |
| MathVista testmini | thinking off, 512 tokens, LLM-extracted | 1,000 (all) | 66.70% | [63.7, 69.6] | 84.50 |
| MathVista testmini | thinking off, 2,048 tokens, LLM-extracted | 1,000 (all) | 73.20% | [70.4, 75.9] | 84.50 |
| MathVista testmini, same items, paired | thinking off 2,048 / **thinking on** 3,072, LLM-extracted | 100 (fixed random subset) | 78.0% / **82.0%** | [73.3, 88.3] for thinking on | 84.50 |

The two subset rows are not full-set numbers and neither belongs next to the card on its own: the
MathVista subset turns out to run 5.33 points easier than the rest of the benchmark. The section
below measures that and says what is left standing once it is accounted for.

CV-Bench by task (thinking off, all items): ADE20K Count 54.97%, ADE20K Relation 90.38%, COCO Count 75.56%,
COCO Relation 95.82%, Omni3D Depth 92.67%, Omni3D Distance 81.17%. Counting is where thinking pays: on the
150-item subset it moves ADE20K Count from the fifties to 66.7% and COCO Count to 91.3%.

MathVista with the rule-based extractor instead of the LLM one: 54.30% / 60.40% / 52.00% for the three rows
above. The model answers with a derivation and the answer is often not the last number, so a regex reads it
wrong; the official protocol uses GPT-4 to extract the answer, and a local Qwen2.5-7B-Instruct with the same
few-shot prompt shape closes most of that gap. Everything here is greedy decoding.

## Is the thinking-on subset representative?

Thinking mode was affordable on a subset only, so the two thinking-on rows rest on 150 and 100 items. A
thinking-on number read off a subset means nothing unless that subset is as hard as the benchmark it stands
for. Scoring the *same* subsets under thinking **off**, where the full set was also measured, answers that
directly (per-item accounting throughout; the 82.42% in the table is the Cambrian mean-of-means, 82.45% here
is the plain per-item rate over the same 2,638 records):

| benchmark | subset, thinking off | remaining items, thinking off | subset is easier by | p |
|---|---|---|---|---|
| CV-Bench | 83.33% (150) | 82.40% (2,488) | **+0.94 points** | 0.77 |
| MathVista | 78.00% (100) | 72.67% (900) | **+5.33 points** | 0.25 |

The CV-Bench subset is representative. **The MathVista subset is not:** it runs 5.33 points easier than the
rest of the benchmark. That is inside sampling noise for n = 100 (p = 0.25, and the draw was a fixed seed, not
a selection), but it is larger than the effect being measured, so the 82.0% on that subset cannot be read as a
full-set number.

What survives the check is the *paired* gain, which is immune to subset difficulty because both arms see the
same items, plus the full-set thinking-off number to carry it onto:

| benchmark | paired gain, on − off | 95% CI | McNemar (exact) | full set, off | **full set, on (estimated)** | card |
|---|---|---|---|---|---|---|
| CV-Bench | +6.00 points | [0.37, 11.63] | b=14, c=5, p=0.064 | 82.45% | **88.45% [82.6, 94.3]** | 86.82 |
| MathVista | +4.00 points | [−3.84, 11.84] | b=10, c=6, p=0.45 | 73.20% | **77.20% [68.9, 85.5]** | 84.50 |

The CV-Bench projection is carried onto the per-item full-set rate, 82.45%, while the card's 86.82 is
presumably in the Cambrian mean-of-means accounting, which gives 82.42% on the same records. The two
differ by 0.03 points here, far inside the interval, but they are not the same quantity.
The projected interval is the full-set thinking-off standard error and the paired-difference
standard error combined in quadrature. Those two are not strictly independent, since the subset is
part of the full set, but at 150 of 2,638 and 100 of 1,000 the overlap term is small.
`scripts/check_readme_numbers.py` re-derives every figure in both tables from `results/`, and
`results/representativeness.json` holds them.

## What is and is not established

- **CV-Bench: reproduced.** The estimated full-set thinking-on accuracy, 88.45% [82.6, 94.3], contains the
  card's 86.82. The paired gain that carries it is +6.00 points, marginal on its own (p = 0.064), and the
  subset it was measured on is representative (+0.94 points). The full-set thinking-off number, 82.42%, is a
  different protocol and should not be read against the card.
- **MathVista: not reproduced at the point estimate, not excluded either.** The estimate is 77.20%
  [68.9, 85.5] against a card value of 84.50. Two things keep this weak in both directions: the thinking-on
  gain is +4.00 points but is not distinguishable from zero on 100 items (p = 0.45), and the subset is 5.33
  points easier than the rest of the benchmark, which is what puts the raw 82.0% so close to the card. Reading
  "82.0% [73.3, 88.3] contains 84.50" as a reproduction would be reading the subset's luck as the model's
  accuracy. The subset is small because thinking-on generation averages 1,400 tokens per item (14 of 100 hit
  the 3,072 cap) and this is one shared GPU.
- Not established: the card's exact prompts, budgets and extractor. The comparison is "does the number survive
  an independent run of the standard protocol", not "is it bit-exact". A larger thinking-on MathVista sample
  would settle the MathVista row; 100 items cannot.

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
python scripts/taichu_summary.py                  # the first table
python scripts/representativeness.py              # the other two
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
