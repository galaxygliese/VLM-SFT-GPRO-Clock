"""Evaluate the base model and every checkpoint of a clock run on the eval split (one vLLM load),
and save thinking / read time / GT time per image and step.

    python tools/eval_clock_checkpoints.py outputs/grpo-clock-think

Writes to <run>/thinking_progress/:
    progress.jsonl   one row per (step, image): step, id, gt, read, err_min, score, thinking, completion
    progress.csv     same without the raw completion
    metrics.csv      per step: exact, within5, hour_correct, median_err, reward, think chars
    report.md        per image: GT, then the thinking + read time at every step
and <run>/checkpoint-N/eval_clock_eval_think.jsonl for each checkpoint (read by tools/show_thinking.py).
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clock_utils import get_prompt, load_split, parse_time, split_thinking, time_score  # noqa: E402


def score(rows, outs, step):
    recs = []
    for r, o in zip(rows, outs):
        text = o.outputs[0].text
        thinking, answer, think_ok = split_thinking(text)
        sc, err = time_score(text, r["hour"], r["minute"])
        pred, _ = parse_time(answer)
        recs.append({"step": step, "id": r["id"], "gt": r["time"], "read": answer.strip()[:50], "err_min": err,
                     "score": sc, "exact": err is not None and err <= 1,
                     "hour_ok": pred is not None and (pred // 60) % 12 == r["hour"] % 12,
                     "think_ok": think_ok, "thinking": thinking, "completion": text})
    return recs


def metrics(recs):
    errs = [x["err_min"] for x in recs if x["err_min"] is not None]
    return {"step": recs[0]["step"], "exact": np.mean([x["exact"] for x in recs]),
            "within5": np.mean([x["err_min"] is not None and x["err_min"] <= 5 for x in recs]),
            "hour_correct": np.mean([x["hour_ok"] for x in recs]),
            "median_err_min": float(np.median(errs)) if errs else float("nan"),
            "reward": np.mean([x["score"] for x in recs]),
            "think_well_formed": np.mean([x["think_ok"] for x in recs]),
            "think_chars": np.mean([len(x["thinking"]) for x in recs])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="e.g. outputs/grpo-clock-think")
    ap.add_argument("--model", default="checkpoints/Qwen3-VL-4B-Instruct-bf16")
    ap.add_argument("--root", default="data/clock")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--no_base", action="store_true")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    ckpts = sorted(glob.glob(os.path.join(args.run, "checkpoint-*")),
                   key=lambda d: int(re.search(r"checkpoint-(\d+)", d).group(1)))
    rows = load_split(f"{args.root}/labels/{args.split}.jsonl", args.root)
    llm = LLM(model=args.model, max_model_len=args.max_tokens + 2048, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"image": 1}, enable_lora=True, max_lora_rank=128, max_loras=1,
              max_num_seqs=32)
    msgs = [[{"role": "user", "content": [
        {"type": "image_pil", "image_pil": Image.open(r["image_path"]).convert("RGB")},
        {"type": "text", "text": get_prompt(True)}]}] for r in rows]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    runs = ([] if args.no_base else [(0, None)]) + \
        [(int(re.search(r"checkpoint-(\d+)", d).group(1)), d) for d in ckpts]
    all_recs, all_metrics = [], []
    for i, (step, d) in enumerate(runs):
        lora = LoRARequest(f"ckpt{step}", i + 1, os.path.abspath(d)) if d else None
        recs = score(rows, llm.chat(msgs, sp, lora_request=lora, use_tqdm=False), step)
        m = metrics(recs)
        all_recs += recs
        all_metrics.append(m)
        print(f"step {step:4d}: exact {m['exact']:.0%}  within5 {m['within5']:.0%}  hour {m['hour_correct']:.0%}  "
              f"median err {m['median_err_min']:.0f} min  think {m['think_chars']:.0f} chars")
        if d:
            with open(os.path.join(d, f"eval_clock_{args.split}_think.jsonl"), "w") as f:
                for x in recs:
                    f.write(json.dumps(x) + "\n")

    out = os.path.join(args.run, "thinking_progress")
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/progress.jsonl", "w") as f:
        for x in all_recs:
            f.write(json.dumps(x) + "\n")
    cols = ["step", "id", "gt", "read", "err_min", "score", "exact", "hour_ok", "think_ok", "thinking"]
    with open(f"{out}/progress.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_recs)
    with open(f"{out}/metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_metrics[0]))
        w.writeheader()
        w.writerows(all_metrics)

    steps = [m["step"] for m in all_metrics]
    lines = [f"# Clock thinking progress: {args.run}", "",
             f"Eval split ({len(rows)} images), greedy decoding. Step 0 = base model.", "",
             "| step | exact (±1 min) | within 5 min | hour correct | median err (min) | think chars |",
             "|---:|---:|---:|---:|---:|---:|"]
    for m in all_metrics:
        lines.append(f"| {m['step']} | {m['exact']:.0%} | {m['within5']:.0%} | {m['hour_correct']:.0%} | "
                     f"{m['median_err_min']:.0f} | {m['think_chars']:.0f} |")
    lines += ["", "## Read time per image", "",
              "| image | GT | " + " | ".join(f"s{s}" for s in steps) + " |",
              "|---|---|" + "---|" * len(steps)]
    by = {(x["step"], x["id"]): x for x in all_recs}
    for r in rows:
        cells = []
        for s in steps:
            x = by[(s, r["id"])]
            cells.append(f"**{x['read']}**" if x["exact"] else (x["read"] or "?"))
        lines.append(f"| {r['id']} | {r['time']} | " + " | ".join(cells) + " |")
    lines += ["", "Bold = exact (±1 min).", "", "## Thinking per image", ""]
    for r in rows:
        lines += [f"### {r['id']}: GT {r['time']}", "", f"![]({os.path.relpath(r['image_path'], out)})", ""]
        for s in steps:
            x = by[(s, r["id"])]
            mark = "✅" if x["exact"] else (f"❌ off by {x['err_min']} min" if x["err_min"] is not None else "❌ unparsed")
            th = " ".join(x["thinking"].split()) or "(no thinking)"
            lines.append(f"- **step {s}**: read **{x['read'] or '?'}** {mark}  \n  {th}")
        lines.append("")
    with open(f"{out}/report.md", "w") as f:
        f.write("\n".join(lines))
    print(f"-> {out}/ (progress.jsonl, progress.csv, metrics.csv, report.md)")


if __name__ == "__main__":
    main()
