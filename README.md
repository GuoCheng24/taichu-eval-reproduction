# taichu-eval-reproduction

Re-measuring two model-card numbers of [ZDTaichu5.0-9B](https://huggingface.co/TaichuAI/ZDTaichu5.0-9B)
(Zidong Taichu 5.0, a Qwen3.5 hybrid GatedDeltaNet decoder with a C-RADIOv4-H vision tower) on one RTX 4090:
**CV-Bench** (card: 86.82) and **MathVista testmini** (card: 84.50).

Both numbers reproduce to within the confidence interval of the sample this GPU could afford, **on the
card's protocol** (thinking mode on, answers extracted by an LLM). Off that protocol they do not, and the
whole gap is the protocol, not the weights.

## Numbers

| benchmark | setting | items | accuracy | 95% CI | card |
|---|---|---|---|---|---|
| CV-Bench, Cambrian-style overall = (mean(ADE20K, COCO) + Omni3D) / 2 | thinking off, 32-token budget | 2,638 (all) | 82.42% | [81.0, 83.9] (item-level) | 86.82 |
| CV-Bench, same items, paired | thinking off / **thinking on**, 3,072-token budget | 150 (fixed random subset) | 83.33% / **89.33%** | [83.4, 93.3] for thinking on | 86.82 |
| MathVista testmini | thinking off, 512 tokens, LLM-extracted | 1,000 (all) | 66.70% | [63.7, 69.6] | 84.50 |
| MathVista testmini | thinking off, 2,048 tokens, LLM-extracted | 1,000 (all) | 73.20% | [70.4, 75.9] | 84.50 |
| MathVista testmini, same items, paired | thinking off 2,048 / **thinking on** 3,072, LLM-extracted | 100 (fixed random subset) | 78.0% / **82.0%** | [73.3, 88.3] for thinking on | 84.50 |

CV-Bench by task (thinking off, all items): ADE20K Count 54.97%, ADE20K Relation 90.38%, COCO Count 75.56%,
COCO Relation 95.82%, Omni3D Depth 92.67%, Omni3D Distance 81.17%. Counting is where thinking pays: on the
150-item subset it moves ADE20K Count from the fifties to 66.7% and COCO Count to 91.3%.

MathVista with the rule-based extractor instead of the LLM one: 54.30% / 60.40% / 52.00% for the three rows
above. The model answers with a derivation and the answer is often not the last number, so a regex reads it
wrong; the official protocol uses GPT-4 to extract the answer, and a local Qwen2.5-7B-Instruct with the same
few-shot prompt shape closes most of that gap. Everything here is greedy decoding.

## What is and is not established

- The card's CV-Bench number is consistent with thinking mode: 89.33% [83.4, 93.3] on 150 items contains 86.82.
  The full-set thinking-off number, 82.42%, is a different protocol and should not be read against the card.
- The card's MathVista number is consistent with thinking mode plus LLM extraction: 82.0% [73.3, 88.3] on 100
  items contains 84.50. The subset is small because thinking-on generation averages 1,400 tokens per item
  (14 of 100 hit the 3,072 cap) and this is one shared GPU.
- Not established: the card's exact prompts, budgets and extractor. The comparison is "does the number survive
  an independent run of the standard protocol", not "is it bit-exact".

## Running it

```bash
pip install timm                                  # the C-RADIO tower needs it
python scripts/taichu_eval.py --bench cvbench   --thinking off --max_new 32
python scripts/taichu_eval.py --bench mathvista --thinking off --max_new 2048 --tag b2048
python scripts/taichu_eval.py --bench cvbench   --thinking on  --max_new 3072 --subset 150
python scripts/taichu_eval.py --bench mathvista --thinking on  --max_new 3072 --subset 100
python scripts/llm_extract.py results/mathvista_think_off_b2048.jsonl   # Qwen2.5-7B-Instruct via vLLM
python scripts/taichu_summary.py
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
