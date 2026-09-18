"""Summarise the Taichu re-measurement: CV-Bench (2D/3D/overall, per source and task), MathVista testmini
(overall, by question/answer type), Wilson 95% CIs, unparsed rates, and the thinking-on vs off paired
delta on the fixed subsets. Model-card claims: CV-Bench 86.82, MathVista-mini 84.50."""

import os as _os

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import collections
import json
import math
import os

T = _os.path.join(_ROOT, "results")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    d = 1 + z * z / n
    return 100 * (c - h) / d, 100 * (c + h) / d


def load(f):
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else []


def acc(rs):
    k = sum(r["ok"] for r in rs)
    n = len(rs)
    lo, hi = wilson(k, n)
    return f"{100 * k / max(1, n):6.2f}% [{lo:.1f}, {hi:.1f}] n={n}"


for mode in ["off", "on_sub150"]:
    f = f"{T}/cvbench_think_{'off' if mode == 'off' else 'on_sub150'}.jsonl"
    rs = load(f)
    if not rs:
        continue
    print(
        f"### CV-Bench, thinking {mode}: {len(rs)} items, unparsed {sum(r['pred'] is None for r in rs)}, card 86.82"
    )
    s2 = [r for r in rs if r["split"] == "test_2d"]
    s3 = [r for r in rs if r["split"] == "test_3d"]
    a2 = sum(r["ok"] for r in s2) / max(1, len(s2))
    a3 = sum(r["ok"] for r in s3) / max(1, len(s3))
    srcacc = lambda name: (
        sum(r["ok"] for r in rs if r["source"] == name)
        / max(1, sum(1 for r in rs if r["source"] == name))
    )
    a2c = (srcacc("ADE20K") + srcacc("COCO")) / 2
    print(
        f"  overall {acc(rs)}   2D {acc(s2)}   3D {acc(s3)}   Cambrian-style: 2D=(ADE+COCO)/2={100 * a2c:.2f}%, overall=(2D+3D)/2={100 * (a2c + a3) / 2:.2f}%"
    )
    by = collections.defaultdict(list)
    for r in rs:
        by[(r["split"], r["source"], r["task"])].append(r)
    for k, v in sorted(by.items()):
        print(f"    {k[0]} {k[1]:<8} {k[2]:<10} {acc(v)}")
    if mode != "off":
        off = {r["id"]: r for r in load(f"{T}/cvbench_think_off.jsonl")}
        pair = [(r["ok"], off[r["id"]]["ok"]) for r in rs if r["id"] in off]
        if pair:
            print(
                f"  paired on the same {len(pair)} items: on {100 * sum(a for a, _ in pair) / len(pair):.2f}%  off {100 * sum(b for _, b in pair) / len(pair):.2f}%  delta {100 * (sum(a for a, _ in pair) - sum(b for _, b in pair)) / len(pair):+.2f}"
            )
        print(
            f"  thinking-on generation: mean {sum(r['ntok'] for r in rs) / len(rs):.0f} tokens, hit cap {sum(r['ntok'] >= 3072 for r in rs)}, closed </think> {sum(r['thought'] for r in rs)}"
        )
for mode in ["off", "on_sub100"]:
    f = f"{T}/mathvista_think_{mode}.jsonl"
    rs = load(f)
    if not rs:
        continue
    print(
        f"### MathVista testmini, thinking {mode}: {len(rs)} items, card 84.50 (official protocol uses an LLM answer extractor)"
    )
    print(f"  overall {acc(rs)}")
    by = collections.defaultdict(list)
    for r in rs:
        by[(r["qtype"], r["atype"])].append(r)
    for k, v in sorted(by.items()):
        print(f"    {k[0]:<12} {k[1]:<8} {acc(v)}")
    if mode != "off":
        off = {r["id"]: r for r in load(f"{T}/mathvista_think_off.jsonl")}
        pair = [(r["ok"], off[r["id"]]["ok"]) for r in rs if r["id"] in off]
        if pair:
            print(
                f"  paired on the same {len(pair)} items: on {100 * sum(a for a, _ in pair) / len(pair):.2f}%  off {100 * sum(b for _, b in pair) / len(pair):.2f}%  delta {100 * (sum(a for a, _ in pair) - sum(b for _, b in pair)) / len(pair):+.2f}"
            )
        print(
            f"  thinking-on generation: mean {sum(r['ntok'] for r in rs) / len(rs):.0f} tokens, hit cap {sum(r['ntok'] >= 3072 for r in rs)}, closed </think> {sum(r['thought'] for r in rs)}"
        )
