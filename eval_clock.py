"""Evaluate clock reading (data/clock) with vLLM, base model or base + LoRA adapter.

Metrics: exact (+-1 min), within 5 min, hour correct, median dial error (min), mean reward,
parse rate, thinking well-formed rate. Per-image thinking/answers are saved next to the adapter.

    python eval_clock.py --thinking
    python eval_clock.py --thinking --adapter outputs/grpo-clock-think/checkpoint-100
"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image

from utils.clock_utils import get_prompt, load_split, parse_time, split_thinking, time_score
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/Qwen3-VL-4B-Instruct-bf16")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--root", default="data/clock")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_split(f"{args.root}/labels/{args.split}.jsonl", args.root)
    llm = LLM(model=args.model, max_model_len=args.max_tokens + 2048, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"image": 1}, enable_lora=args.adapter is not None, max_lora_rank=128,
              max_num_seqs=32)
    msgs = [[{"role": "user", "content": [
        {"type": "image_pil", "image_pil": Image.open(r["image_path"]).convert("RGB")},
        {"type": "text", "text": get_prompt(args.thinking)}]}] for r in rows]
    lora = LoRARequest("adapter", 1, os.path.abspath(args.adapter)) if args.adapter else None
    t = time.time()
    outs = llm.chat(msgs, SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens),
                    lora_request=lora)
    gen_s = time.time() - t

    recs = []
    for r, o in zip(rows, outs):
        text = o.outputs[0].text
        thinking, answer, think_ok = split_thinking(text)
        score, err = time_score(text, r["hour"], r["minute"])
        pred, strict = parse_time(answer)
        hour_ok = pred is not None and (pred // 60) % 12 == r["hour"] % 12
        recs.append({"id": r["id"], "gt": r["time"], "answer": answer.strip()[:50], "err_min": err,
                     "score": score, "hour_ok": hour_ok, "strict": strict, "think_ok": think_ok,
                     "thinking": thinking, "completion": text})

    errs = [x["err_min"] for x in recs if x["err_min"] is not None]
    m = {
        "exact(+-1min)": np.mean([x["err_min"] is not None and x["err_min"] <= 1 for x in recs]),
        "within5min": np.mean([x["err_min"] is not None and x["err_min"] <= 5 for x in recs]),
        "hour_correct": np.mean([x["hour_ok"] for x in recs]),
        "median_err_min": float(np.median(errs)) if errs else None,
        "reward": np.mean([x["score"] for x in recs]),
        "parsed": len(errs) / len(recs),
        "strict_format": np.mean([x["strict"] for x in recs]),
    }
    if args.thinking:
        m["think_well_formed"] = np.mean([x["think_ok"] for x in recs])
        m["think_chars"] = np.mean([len(x["thinking"]) for x in recs])
    m = {k: (float(v) if v is not None else None) for k, v in m.items()}
    m["gen_seconds"] = gen_s

    print(f"\n{args.model}" + (f" + {args.adapter}" if args.adapter else "")
          + f"  [{args.split}, {len(rows)} images, thinking={args.thinking}]")
    for k, v in m.items():
        print(f"  {k:18s} {v:.3f}" if isinstance(v, float) else f"  {k:18s} {v}")
    print("  per image: " + " ".join(f"{x['gt']}->{x['answer'] or '?'}" for x in recs))

    tag = f"clock_{args.split}{'_think' if args.thinking else ''}"
    out = args.out or (os.path.join(args.adapter, f"eval_{tag}.jsonl") if args.adapter
                       else f"outputs/eval/base_{tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")
    with open(out.replace(".jsonl", "_metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    print("->", out)


if __name__ == "__main__":
    main()
