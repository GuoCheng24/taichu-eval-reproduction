"""One scoring of one unchanged file, printed as `name value` for `groundwork noise`.

The Taichu numbers in this repository depend on an LLM answer-extractor, so the
instrument that produces them is a language model and not a regular expression.
Before any arm-to-arm difference is quoted, the spread of that instrument on a
file that never changes has to be on disk.

Two arms, because they ask different questions:

    --order as-is      re-extract the same prompts in the same order. Identical
                       batching, so this measures process-to-process
                       nondeterminism alone.
    --order shuffled   the same items, permuted. The item set is unchanged and
                       every prompt is scored, but which prompts share a batch
                       is not - and bf16 matrix multiplication is not batch
                       invariant, which this repository's sibling measures
                       directly (github.com/GuoCheng24/batch-logprob-gap).

Usage, through the noise tool so the framing comes with it:

    groundwork noise --n 10 \\
      --command 'python scripts/extractor_noise_probe.py results/FILE.jsonl'

Writes nothing except its own scratch copy, so the committed *_llmext.json
files are never touched.
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="a generations .jsonl under results/")
    ap.add_argument("--order", choices=("as-is", "shuffled"), default="as-is")
    ap.add_argument("--seed", type=int, default=None,
                    help="permutation seed; default is a fresh one each run, "
                         "which is what makes repeated runs a measurement of "
                         "batch composition rather than of one permutation")
    a = ap.parse_args(argv)

    with open(a.file, encoding="utf-8") as fh:
        rows = [json.loads(ln) for ln in fh if ln.strip()]
    if a.order == "shuffled":
        rnd = random.Random(a.seed if a.seed is not None else random.randrange(1 << 30))
        rnd.shuffle(rows)

    tmp = tempfile.mkdtemp(prefix="extnoise_")
    scratch = os.path.join(tmp, "arm.jsonl")
    with open(scratch, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    try:
        p = subprocess.run([sys.executable,
                            os.path.join(ROOT, "scripts", "llm_extract.py"), scratch],
                           capture_output=True, text=True, timeout=3600)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("###")), None)
        if line is None:
            sys.stderr.write(p.stdout[-2000:] + p.stderr[-2000:])
            return 3
        m = re.search(r"rule ([\d.]+)% -> LLM-extracted ([\d.]+)%", line)
        if not m:
            sys.stderr.write(line + "\n")
            return 3
        print(f"rule {m.group(1)}")
        print(f"llm_extracted {m.group(2)}")
        # the number the arms are actually compared on
        print(f"n_items {len(rows)}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
