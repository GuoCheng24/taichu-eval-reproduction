"""MathVista-style answer extraction with a local LLM (the official protocol uses GPT-4 with this few-shot
prompt shape). Re-scores saved Taichu outputs from the tail of the response. argv: files..."""

import os as _os

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import glob
import json
import os
import re
import sys

import pyarrow.parquet as pq

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from vllm import LLM, SamplingParams

HUB = _os.path.expanduser(_os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub"))
meta = {
    r["pid"]: r
    for r in pq.read_table(
        glob.glob(
            f"{HUB}/datasets--AI4Math--MathVista/snapshots/*/data/testmini-*.parquet"
        )[0]
    ).to_pylist()
}
DEMO = """Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?
Model response: The number missing in the sequence is 14.
Extracted answer: 14

Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?
Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.
Extracted answer: 0.6

Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line graph saw its maximum peak?
Model response: The line graph saw its maximum peak between 2007 and 2008.
Extracted answer: [2007, 2008]

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5
Model response: The correct answer is (B) 8/11.
Extracted answer: B

"""


def correct(it, pred):
    g = str(it["answer"]).strip()
    pred = str(pred).strip().strip(".")
    if it["question_type"] == "multi_choice":
        ch = it["choices"] or []
        m = re.match(r"^\(?([A-Z])\)?$", pred)
        if m and ord(m.group(1)) - 65 < len(ch):
            pred = ch[ord(m.group(1)) - 65]
        return pred.strip().lower() == g.lower()
    if it["answer_type"] == "list":
        return re.sub(r"\s+", "", pred) == re.sub(r"\s+", "", g)
    try:
        prec = int(it["precision"] or 0)
        return (
            abs(
                round(float(pred.replace(",", "")), prec)
                - round(float(g.replace(",", "")), prec)
            )
            < 1e-6
        )
    except ValueError:
        return pred.lower() == g.lower()


llm = LLM(
    "Qwen/Qwen2.5-7B-Instruct",
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
    max_model_len=4096,
    enforce_eager=True,
)
tok = llm.get_tokenizer()
for f in sys.argv[1:]:
    rs = [json.loads(l) for l in open(f)]
    prompts = []
    for r in rs:
        it = meta[r["id"]]
        q = it["query"]
        prompts.append(
            tok.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": DEMO
                        + q
                        + "\n\nModel response: "
                        + r["text"][-1200:]
                        + "\n\nExtracted answer: ",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    outs = llm.generate(
        prompts, SamplingParams(temperature=0.0, max_tokens=32), use_tqdm=False
    )
    n_ok = 0
    by = {}
    for r, o in zip(rs, outs):
        it = meta[r["id"]]
        pred = o.outputs[0].text.strip().split("\n")[0]
        ok = correct(it, pred)
        n_ok += ok
        k = f"{it['question_type']}/{it['answer_type']}"
        by.setdefault(k, [0, 0])
        by[k][0] += ok
        by[k][1] += 1
        r["llm_pred"] = pred
        r["llm_ok"] = bool(ok)
    json.dump(rs, open(f.replace(".jsonl", "_llmext.json"), "w"))
    print(
        f"### {os.path.basename(f)}: rule {100 * sum(r['ok'] for r in rs) / len(rs):.2f}% -> LLM-extracted {100 * n_ok / len(rs):.2f}%  "
        + " ".join(f"{k}={100 * v[0] / v[1]:.1f}%" for k, v in sorted(by.items())),
        flush=True,
    )
print("LLMEXT_DONE")
