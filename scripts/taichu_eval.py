"""Independent re-measurement of two ZDTaichu5.0-9B model-card numbers on one RTX 4090:
CV-Bench (card: 86.82) and MathVista testmini (card: 84.50). Greedy decoding, the benchmark's own
prompt text, rule-based answer extraction (the official MathVista protocol uses an LLM extractor,
so the free-form part here is a lower bound). Resumable: one JSONL line per item."""

import os as _os

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import argparse
import glob
import io
import json
import os
import re
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import pyarrow.parquet as pq
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--bench", choices=["cvbench", "mathvista"], required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument(
    "--subset",
    type=int,
    default=0,
    help="evaluate a fixed random subset of this size (seed 0)",
)
ap.add_argument("--tag", default="", help="suffix for the output file name")
ap.add_argument("--out", default=_os.path.join(_ROOT, "results"))
ap.add_argument("--max_new", type=int, default=0)
ap.add_argument("--thinking", choices=["default", "off", "on"], default="default")
ap.add_argument(
    "--shard",
    default="",
    metavar="I/N",
    help="evaluate only every Nth item starting at I, so one arm can be spread over several "
         "idle GPUs. Each shard writes its own file (use --tag) and the shards are concatenated "
         "afterwards. Decoding is one item at a time, so which GPU takes which item changes "
         "nothing about the result - but the split is recorded rather than assumed harmless.",
)
a = ap.parse_args()
HUB = os.path.expanduser("~/.cache/huggingface/hub")
MODEL = glob.glob(f"{HUB}/models--TaichuAI--ZDTaichu5.0-9B/snapshots/*/")[0]
IMG = f"{a.out}/img"
os.makedirs(IMG, exist_ok=True)
outf = f"{a.out}/{a.bench}{'_think_' + a.thinking if a.thinking != 'default' else ''}{'_sub' + str(a.subset) if a.subset else ''}{'_' + a.tag if a.tag else ''}{'_smoke' if a.limit else ''}.jsonl"
done = set()
if os.path.exists(outf):
    with open(outf) as fh:
        done = {json.loads(line)["id"] for line in fh}


def running_acc(path):
    """Accuracy over everything written so far, in one pass over the file."""
    hit = total = 0
    with open(path) as fh:
        for line in fh:
            hit += json.loads(line)["ok"]
            total += 1
    return hit / max(1, total)


def load_items():
    items = []
    if a.bench == "cvbench":
        base = glob.glob(f"{HUB}/datasets--nyu-visionx--CV-Bench/snapshots/*/")[0]
        for split in ["test_2d", "test_3d"]:
            t = pq.read_table(f"{base}/{split}.parquet").to_pylist()
            for r in t:
                items.append(
                    {
                        "id": f"{split}:{r['idx']}",
                        "split": split,
                        "task": r["task"],
                        "source": r["source"],
                        "prompt": r["prompt"],
                        "answer": r["answer"],
                        "choices": r["choices"],
                        "img": r["image"]["bytes"],
                    }
                )
    else:
        base = glob.glob(f"{HUB}/datasets--AI4Math--MathVista/snapshots/*/")[0]
        t = pq.read_table(glob.glob(f"{base}/data/testmini-*.parquet")[0]).to_pylist()
        for r in t:
            items.append(
                {
                    "id": r["pid"],
                    "prompt": r["query"],
                    "answer": r["answer"],
                    "choices": r["choices"],
                    "question_type": r["question_type"],
                    "answer_type": r["answer_type"],
                    "precision": r["precision"],
                    "img": r["decoded_image"]["bytes"],
                }
            )
    if a.subset:
        import random

        rng = random.Random(0)
        items = rng.sample(items, a.subset)
    return items[: a.limit] if a.limit else items


def letter_of(text):
    m = (
        re.search(r"\(([A-F])\)", text)
        or re.search(r"\b([A-F])\b", text.strip()[:8])
        or re.search(r"(?:answer|Answer)[^A-F]{0,20}\b([A-F])\b", text)
    )
    return m.group(1) if m else None


def score_mathvista(it, text):
    text = text.strip()
    final = text
    m = re.search(r"(?:answer|Answer)\s*(?:is|:)?\s*(.+)$", text, re.DOTALL)
    if m:
        final = m.group(1).strip()
    if it["question_type"] == "multi_choice":
        ch = it["choices"] or []
        # letter first, then exact choice text
        lm = re.search(r"\(([A-Z])\)", final) or re.search(r"^\s*([A-Z])\b", final)
        if lm and ord(lm.group(1)) - 65 < len(ch):
            pred = ch[ord(lm.group(1)) - 65]
        else:
            hits = [c for c in ch if c.strip().lower() in final.lower()]
            pred = max(hits, key=len) if hits else final
        return pred.strip().lower() == str(it["answer"]).strip().lower(), pred
    if it["answer_type"] == "list":
        norm = lambda x: re.sub(r"\s+", "", str(x))
        m2 = re.search(r"\[[^\]]*\]", final)
        return (norm(m2.group(0)) == norm(it["answer"])) if m2 else False, (
            m2.group(0) if m2 else final
        )
    nums = re.findall(r"-?\d+(?:\.\d+)?", final.replace(",", ""))
    if not nums:
        return False, final
    pred = nums[-1]
    try:
        p, g = float(pred), float(str(it["answer"]).replace(",", ""))
        prec = it["precision"] if it["precision"] is not None else 0
        return abs(round(p, int(prec)) - round(g, int(prec))) < 1e-6, pred
    except ValueError:
        return pred.strip().lower() == str(it["answer"]).strip().lower(), pred


items = load_items()
if a.shard:
    try:
        _i, _n = (int(x) for x in a.shard.split("/"))
    except ValueError:
        sys.exit(f"--shard wants I/N, got {a.shard!r}")
    if not (_n >= 1 and 0 <= _i < _n):
        sys.exit(f"--shard {a.shard}: need 0 <= I < N and N >= 1")
    # Sliced BEFORE the already-done filter, so a shard owns the same items for
    # the life of the run. Slicing the filtered list instead makes ownership
    # depend on how far each shard happens to have got, and two shards restarted
    # at different points then take the same item while another is taken by
    # nobody.
    items = items[_i::_n]
    print(f"{a.bench}: shard {_i}/{_n}, {len(items)} items owned", flush=True)
items = [it for it in items if it["id"] not in done]
print(f"{a.bench}: {len(items)} to do ({len(done)} already)", flush=True)
if not items:
    sys.exit()
from transformers import AutoConfig, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, use_fast=False)
cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
cls = get_class_from_dynamic_module(cfg.auto_map["AutoModel"], MODEL)
# the shipped code declares this as a list; transformers 5.16 unions it with a set
for klass in [cls] + [
    c for c in cls.__mro__ if hasattr(c, "_keys_to_ignore_on_load_unexpected")
]:
    v = getattr(klass, "_keys_to_ignore_on_load_unexpected", None)
    if isinstance(v, list):
        klass._keys_to_ignore_on_load_unexpected = set(v)
# the shipped configs default to flash_attention_2, which is not installed here: force sdpa on every sub-config
# the Qwen3.5 decoder takes sdpa; the C-RADIO vision tower only implements eager (and flash-attn)
ATTN = {"": "sdpa", "llm_config": "sdpa", "vision_config": "eager"}
for name, impl in ATTN.items():
    c = cfg if name == "" else getattr(cfg, name, None)
    if c is None:
        continue
    # transformers has renamed this field twice, and the shipped config classes accept a
    # different subset of the three on each version, so set all three and let the ones this
    # version rejects raise. Writing only the attributes that already exist is not the same
    # thing: on some versions the field this loop has to create is the one that is read.
    for k in (
        "_attn_implementation",
        "attn_implementation",
        "_attn_implementation_internal",
    ):
        try:
            setattr(c, k, impl)
        except (AttributeError, TypeError, ValueError):
            pass
model = cls.from_pretrained(
    MODEL,
    config=cfg,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation=ATTN,
).eval()
print(
    "attn:",
    getattr(cfg, "_attn_implementation", None),
    {
        k: getattr(getattr(cfg, k, None), "_attn_implementation", None)
        for k in ("llm_config", "vision_config")
    },
    flush=True,
)
MAXNEW = a.max_new or (32 if a.bench == "cvbench" else 1024)
t0 = time.time()
n = 0
with open(outf, "a") as fh:
    for it in items:
        path = f"{IMG}/{a.bench}_{it['id'].replace(':', '_')}.png"
        if not os.path.exists(path):
            Image.open(io.BytesIO(it["img"])).convert("RGB").save(path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": path},
                    {"type": "text", "text": it["prompt"]},
                ],
            }
        ]
        if a.thinking == "default":
            inputs = processor.from_messages(messages, return_tensors="pt").to(
                model.device
            )
        else:
            # replicate from_messages, but pass enable_thinking to the chat template
            import importlib

            pv = importlib.import_module(
                type(processor).__module__.rsplit(".", 1)[0] + ".vision_utils"
            ).process_vision_info
            image_inputs, video_inputs, video_kwargs = pv(messages)
            prompt = processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=(a.thinking == "on"),
            )
            inputs = processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                **(video_kwargs or {}),
            ).to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=MAXNEW, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1] :]
        raw = processor.batch_decode(gen, skip_special_tokens=False)[0]
        text = processor.batch_decode(gen, skip_special_tokens=True)[0]
        thought = "</think>" in raw
        final = raw.split("</think>")[-1] if thought else text
        for tokstr in ("<|im_end|>", "<|endoftext|>", "<think>", "</think>"):
            final = final.replace(tokstr, "")
        final = final.strip() or text[-300:]
        ntok = int(gen.shape[1])
        if a.bench == "cvbench":
            pred = letter_of(final)
            gold = re.sub(r"[()]", "", it["answer"]).strip()
            ok = pred == gold
            rec = {
                "id": it["id"],
                "split": it["split"],
                "task": it["task"],
                "source": it["source"],
                "gold": gold,
                "pred": pred,
                "ok": bool(ok),
                "thought": thought,
                "ntok": ntok,
                "final": final,
                "text": text[-800:],
            }
        else:
            ok, pred = score_mathvista(it, final)
            rec = {
                "id": it["id"],
                "qtype": it["question_type"],
                "atype": it["answer_type"],
                "gold": str(it["answer"]),
                "pred": str(pred)[:80],
                "ok": bool(ok),
                "thought": thought,
                "ntok": ntok,
                "final": final,
                "text": text[-800:],
            }
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        n += 1
        if n % 25 == 0 or n == len(items):
            print(
                f"  {n}/{len(items)}  {(time.time() - t0) / n:.1f}s/item  acc so far {running_acc(outf):.3f}",
                flush=True,
            )
print("EVAL_DONE", flush=True)
