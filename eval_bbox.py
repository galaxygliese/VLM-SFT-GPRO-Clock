"""Evaluate a Qwen3-VL model (optionally + LoRA adapter) on the bbox task with vLLM.

Metrics are against the SAM3 pseudo-labels, per class and overall, at IoU >= 0.5:
precision / recall / F1 (micro over boxes), soft-F1 (= the GRPO detection reward),
strict-JSON rate and truncation rate. VLM outputs carry no confidence scores, so there is no mAP.

    python eval_bbox.py                                              # base model
    python eval_bbox.py --adapter outputs/grpo-bbox/checkpoint-200   # base + LoRA
    python eval_bbox.py --thinking --adapter outputs/grpo-bbox-think/checkpoint-100
"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image

from bbox_utils import CLASSES, detection_score, get_prompt, load_split, match, parse_objects, prf, split_thinking


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/Qwen3-VL-4B-Instruct-bf16")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (PEFT format)")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--root", default="data/fisheye8k")
    ap.add_argument("--max_tokens", type=int, default=3072)
    ap.add_argument("--thinking", action="store_true", help="use the <thinking> prompt; score only the JSON after it")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None, help="only the first N images")
    ap.add_argument("--out", default=None, help="predictions jsonl (default: next to adapter or outputs/eval/)")
    ap.add_argument("--wandb", action="store_true", help="log metrics to wandb project vlm-grpo-bbox")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows = load_split(f"{args.root}/labels/{args.split}.jsonl")[:args.limit]
    llm = LLM(model=args.model, max_model_len=args.max_tokens + 2048, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"image": 1}, enable_lora=args.adapter is not None, max_lora_rank=128,
              max_num_seqs=16)
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    msgs = [[{"role": "user", "content": [
        {"type": "image_pil", "image_pil": Image.open(f"{args.root}/{r['image']}").convert("RGB")},
        {"type": "text", "text": get_prompt(args.thinking)}]}] for r in rows]
    lora = LoRARequest("adapter", 1, os.path.abspath(args.adapter)) if args.adapter else None
    t = time.time()
    outs = llm.chat(msgs, sp, lora_request=lora)
    gen_s = time.time() - t

    agg = {c: np.zeros(4) for c in CLASSES}
    soft, strict, trunc, ntok, think_ok, think_len = [], [], [], [], [], []
    preds = []
    for r, o in zip(rows, outs):
        text = o.outputs[0].text
        thinking, answer, tok = split_thinking(text)
        think_ok.append(tok)
        think_len.append(len(thinking))
        pred, ok = parse_objects(answer)
        for c, v in match(pred, r["objects"]).items():
            agg[c] += v
        soft.append(detection_score(pred, r["objects"]))
        strict.append(ok)
        trunc.append(o.outputs[0].finish_reason == "length")
        ntok.append(len(o.outputs[0].token_ids))
        preds.append({"image": r["image"], "pred": pred, "thinking": thinking, "raw": text})

    metrics = {}
    print(f"\n{args.model}" + (f" + {args.adapter}" if args.adapter else "") + f"  [{args.split}, {len(rows)} images]")
    print(f"{'class':14s} {'n_gt':>5s} {'n_pred':>6s} {'P':>6s} {'R':>6s} {'F1':>6s} {'mIoU(tp)':>8s}")
    tot = np.zeros(4)
    for c in CLASSES:
        tp, npred, ngt, s = agg[c]
        tot += agg[c]
        p, rcl, f = prf(tp, npred, ngt)
        miou = s / tp if tp else 0.0
        metrics.update({f"{c}/P": p, f"{c}/R": rcl, f"{c}/F1": f})
        print(f"{c:14s} {int(ngt):5d} {int(npred):6d} {p:6.3f} {rcl:6.3f} {f:6.3f} {miou:8.3f}")
    p, rcl, f = prf(tot[0], tot[1], tot[2])
    print(f"{'ALL (micro)':14s} {int(tot[2]):5d} {int(tot[1]):6d} {p:6.3f} {rcl:6.3f} {f:6.3f} {tot[3] / max(tot[0], 1):8.3f}")
    macro_f1 = float(np.mean([metrics[f"{c}/F1"] for c in CLASSES]))
    metrics.update({"P": p, "R": rcl, "F1": f, "macro_F1": macro_f1, "soft_F1": float(np.mean(soft)),
                    "strict_json": float(np.mean(strict)), "truncated": float(np.mean(trunc)),
                    "completion_tokens": float(np.mean(ntok)), "gen_seconds": gen_s})
    if args.thinking:
        metrics.update({"think_well_formed": float(np.mean(think_ok)), "think_chars": float(np.mean(think_len))})
        print(f"thinking well-formed {np.mean(think_ok):.0%} | thinking {np.mean(think_len):.0f} chars")
    print(f"macro F1 {macro_f1:.3f} | soft-F1 (reward) {np.mean(soft):.3f} | strict JSON {np.mean(strict):.0%} "
          f"| truncated {np.mean(trunc):.0%} | {np.mean(ntok):.0f} tok/answer | {gen_s:.0f}s")

    tag = f"{args.split}{'_think' if args.thinking else ''}"
    out = args.out or (os.path.join(args.adapter, f"eval_{tag}.jsonl") if args.adapter
                       else f"outputs/eval/base_{tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f_:
        for p_ in preds:
            f_.write(json.dumps(p_) + "\n")
    with open(out.replace(".jsonl", "_metrics.json"), "w") as f_:
        json.dump(metrics, f_, indent=2)
    print("predictions ->", out)

    if args.wandb:
        import wandb
        wandb.init(project="vlm-grpo-bbox", job_type="eval",
                   name=f"eval-{os.path.basename(args.adapter.rstrip('/')) if args.adapter else 'base'}",
                   config=vars(args))
        wandb.log({f"eval/{k}": v for k, v in metrics.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
