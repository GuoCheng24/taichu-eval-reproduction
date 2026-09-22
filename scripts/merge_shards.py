#!/usr/bin/env python
"""Merge the per-shard output files of one arm back into a single file.

An arm may be spread over several idle GPUs with `--shard I/N`. Each shard owns
`items[I::N]` of the fixed subset and writes its own file, and each shard file is
pre-seeded with whatever the un-sharded run had already produced so that no item
is generated twice. That pre-seeding means the same id legitimately appears in
more than one shard file, so merging has to dedupe - and a duplicate whose rows
disagree is a real problem, not a tidying job, so it fails instead.

    python scripts/merge_shards.py --out results/mathvista_think_on_sub100_TA3072.jsonl \
        --shards results/mathvista_think_on_sub100_TA3072_s*.jsonl --expect 100
"""
import argparse, json, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--expect", type=int, default=0, help="required number of distinct ids")
    args = ap.parse_args()

    rows, seen, clashes = {}, {}, []
    for path in args.shards:
        try:
            fh = open(path, encoding="utf-8")
        except FileNotFoundError:
            print(f"  {path}: missing, skipped")
            continue
        n = 0
        with fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                n += 1
                key = r["id"]
                if key in rows and rows[key] != r:
                    differing = sorted(k for k in set(rows[key]) | set(r)
                                       if rows[key].get(k) != r.get(k))
                    clashes.append((key, seen[key], path, differing))
                rows[key] = r
                seen[key] = path
        print(f"  {path}: {n} rows")

    if clashes:
        print(f"\n{len(clashes)} id(s) appear in more than one shard with different content:")
        for key, a, b, fields in clashes[:10]:
            print(f"  {key}: {a} vs {b} differ in {fields}")
        return 1

    if args.expect and len(rows) != args.expect:
        print(f"\n{len(rows)} distinct ids, expected {args.expect} - the shards do not cover the subset")
        return 1

    order = sorted(rows, key=lambda k: (str(type(k)), k))
    with open(args.out, "w", encoding="utf-8") as fh:
        for k in order:
            fh.write(json.dumps(rows[k], ensure_ascii=False) + "\n")
    closed = sum(1 for r in rows.values() if r.get("thought"))
    at_cap = sorted(r["ntok"] for r in rows.values())[-3:]
    print(f"\nwrote {args.out}: {len(rows)} items, "
          f"{len(rows) - closed} with an unclosed reasoning block, "
          f"largest ntok {at_cap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
