"""Show how the thinking on the same images changes across checkpoints.

Reads, per step:
  <run>/thinking_probes.jsonl           (written during training by ThinkingProbeCallback, incl. step 0)
  <run>/checkpoint-N/thinking_probe.jsonl
  <run>/checkpoint-N/eval_*think*.jsonl (from eval_bbox.py / eval_clock.py --thinking --adapter ...)
and optionally a base-model eval file (--base) as step 0.

    python tools/show_thinking.py outputs/grpo-clock-think                 # summary + first probe image
    python tools/show_thinking.py outputs/grpo-clock-think --id eval/clock_003.jpg --chars 0
    python tools/show_thinking.py outputs/grpo-bbox-think --base outputs/eval/base_eval_think.jsonl --all
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict


def read_jsonl(p):
    return [json.loads(l) for l in open(p)]


def collect(run, base=None):
    by_step = defaultdict(dict)  # step -> id -> record
    if base:
        for r in read_jsonl(base):
            by_step[0].setdefault(r.get("id") or r.get("image"), r)
    p = os.path.join(run, "thinking_probes.jsonl")
    if os.path.exists(p):
        for r in read_jsonl(p):
            by_step[r["step"]][r["id"]] = r
    for d in glob.glob(os.path.join(run, "checkpoint-*")):
        step = int(re.search(r"checkpoint-(\d+)", d).group(1))
        files = [os.path.join(d, "thinking_probe.jsonl")] + sorted(glob.glob(os.path.join(d, "eval_*think*.jsonl")))
        for f in files:
            if os.path.exists(f) and not f.endswith("_metrics.json"):
                for r in read_jsonl(f):
                    by_step[step].setdefault(r.get("id") or r.get("image"), r)
    return dict(sorted(by_step.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="training output dir, e.g. outputs/grpo-clock-think")
    ap.add_argument("--base", default=None, help="base-model eval jsonl to show as step 0")
    ap.add_argument("--id", default=None, help="image id to show (default: first probe image)")
    ap.add_argument("--all", action="store_true", help="show every image that appears at every step")
    ap.add_argument("--chars", type=int, default=600, help="truncate thinking (0 = full)")
    args = ap.parse_args()

    data = collect(args.run, args.base)
    if not data:
        raise SystemExit(f"no thinking records under {args.run}")

    print(f"{'step':>6s} {'n':>3s} {'mean score':>10s} {'think chars':>11s}")
    for step, recs in data.items():
        rs = list(recs.values())
        sc = [r["score"] for r in rs if r.get("score") is not None]
        tc = [len(r.get("thinking") or "") for r in rs]
        print(f"{step:6d} {len(rs):3d} {sum(sc) / max(len(sc), 1):10.3f} {sum(tc) / max(len(tc), 1):11.0f}")

    common = set.intersection(*(set(r) for r in data.values()))
    ids = sorted(common) if args.all else [args.id or (sorted(common)[0] if common else next(iter(data[min(data)])))]
    for i in ids:
        print("\n" + "=" * 100 + f"\n{i}")
        for step, recs in data.items():
            r = recs.get(i)
            if r is None:
                continue
            th = r.get("thinking") or ""
            if args.chars and len(th) > args.chars:
                th = th[:args.chars] + " ..."
            gt = f" gt={r['gt']}" if "gt" in r else ""
            ans = r.get("answer") or (f"{r['n_pred']} boxes" if "n_pred" in r else "")
            print(f"\n--- step {step} | score {r.get('score', float('nan')):.3f}{gt} | answer: {ans}\n{th}")


if __name__ == "__main__":
    main()
