"""Tail-first rule-based re-scoring of saved MathVista outputs: the answer is at the end of the response,
so search the last 250 characters, prefer 'Answer: ...' patterns and the LAST option letter, and match option
text only in that tail. Reports the old score, the new score, and the changed items."""

import os as _os

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import collections
import glob
import json
import re
import sys

import pyarrow.parquet as pq

HUB = _os.path.expanduser(_os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub"))
meta = {
    r["pid"]: r
    for r in pq.read_table(
        glob.glob(
            f"{HUB}/datasets--AI4Math--MathVista/snapshots/*/data/testmini-*.parquet"
        )[0]
    ).to_pylist()
}


def tail_extract(it, text):
    t = text.strip()
    tail = t[-250:]
    m = re.findall(
        r"(?:[Aa]nswer|ANSWER)\s*(?:is|:)?\s*\**\s*(.+?)(?:\.\s*$|$|\n)", tail
    )
    cand = m[-1].strip(" *.") if m else tail
    ch = it["choices"] or []
    if it["question_type"] == "multi_choice":
        letters = re.findall(r"\(([A-Z])\)", cand) or re.findall(
            r"\b([A-Z])\b(?=[\s.:)]|$)", cand[:6]
        )
        if letters and ord(letters[-1]) - 65 < len(ch):
            return ch[ord(letters[-1]) - 65]
        hits = [c for c in ch if c.strip().lower() in cand.lower()]
        if not hits:
            hits = [c for c in ch if c.strip().lower() in tail.lower()]
        if hits:
            # the option mentioned LAST in the tail wins
            return max(hits, key=lambda c: tail.lower().rfind(c.strip().lower()))
        return cand
    if it["answer_type"] == "list":
        m2 = re.search(r"\[[^\]]*\]", cand) or re.search(r"\[[^\]]*\]", tail)
        return m2.group(0) if m2 else cand
    nums = re.findall(r"-?\d+(?:\.\d+)?", cand.replace(",", "")) or re.findall(
        r"-?\d+(?:\.\d+)?", tail.replace(",", "")
    )
    return nums[-1] if nums else cand


def correct(it, pred):
    g = str(it["answer"]).strip()
    if it["question_type"] == "multi_choice":
        return pred.strip().lower() == g.lower()
    if it["answer_type"] == "list":
        return re.sub(r"\s+", "", pred) == re.sub(r"\s+", "", g)
    try:
        prec = int(it["precision"] or 0)
        return (
            abs(round(float(pred), prec) - round(float(g.replace(",", "")), prec))
            < 1e-6
        )
    except ValueError:
        return pred.strip().lower() == g.lower()


for f in sys.argv[1:]:
    rs = [json.loads(l) for l in open(f)]
    old = sum(r["ok"] for r in rs)
    new = 0
    by = collections.defaultdict(lambda: [0, 0])
    flips = []
    for r in rs:
        it = meta[r["id"]]
        pred = tail_extract(it, r["text"])
        ok = correct(it, pred)
        new += ok
        k = (it["question_type"], it["answer_type"])
        by[k][0] += ok
        by[k][1] += 1
        if ok != r["ok"] and len(flips) < 6:
            flips.append((r["id"], r["ok"], ok, str(it["answer"])[:20], str(pred)[:30]))
    print(
        f"{f.split('/')[-1]}: old {100 * old / len(rs):.2f}% -> tail-first {100 * new / len(rs):.2f}%  "
        + " ".join(
            f"{k[0][:5]}/{k[1]}={100 * v[0] / v[1]:.1f}%" for k, v in sorted(by.items())
        )
    )
    for x in flips:
        print("   flip:", x)
