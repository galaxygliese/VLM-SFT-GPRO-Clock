"""Compare clock-reading models on the eval split in one vLLM load: the base model + any LoRA adapters.

    python tools/eval_clock_compare.py \\
        sft=outputs/sft-clock-think/final \\
        grpo=outputs/grpo-clock-think-mix/final \\
        sft_grpo=outputs/grpo-clock-think-sft/final

Each positional argument is NAME=ADAPTER_DIR ("base" is always evaluated first unless --no_base).
Greedy decoding, thinking prompt, same scoring as tools/eval_clock_checkpoints.py.

Writes to --out (default outputs/compare_clock/):
    metrics.csv    one row per model: exact, within5, hour_correct, median_err_min, reward, think stats
    per_image.csv  one row per (model, image): gt, read time, error, thinking
    report.md      metrics table + read time per image for every model
"""
import argparse
import csv
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clock_utils import get_prompt, load_split  # noqa: E402
from eval_clock_checkpoints import metrics, score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapters", nargs="*", help="NAME=ADAPTER_DIR")
    ap.add_argument("--model", default="checkpoints/Qwen3-VL-4B-Instruct-bf16")
    ap.add_argument("--root", default="data/clock")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--no_base", action="store_true")
    ap.add_argument("--out", default="outputs/compare_clock")
    args = ap.parse_args()

    runs = [] if args.no_base else [("base", None)]
    for a in args.adapters:
        name, _, path = a.partition("=")
        if not path or not os.path.isfile(os.path.join(path, "adapter_config.json")):
            sys.exit(f"bad adapter '{a}': expected NAME=DIR with DIR/adapter_config.json")
        runs.append((name, path))

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows = load_split(f"{args.root}/labels/{args.split}.jsonl", args.root)
    llm = LLM(model=args.model, max_model_len=args.max_tokens + 2048, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"image": 1}, enable_lora=True, max_lora_rank=128, max_loras=1,
              max_num_seqs=32)
    msgs = [[{"role": "user", "content": [
        {"type": "image_pil", "image_pil": Image.open(r["image_path"]).convert("RGB")},
        {"type": "text", "text": get_prompt(True)}]}] for r in rows]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    all_recs, all_m = [], []
    for i, (name, path) in enumerate(runs):
        lora = LoRARequest(name, i + 1, os.path.abspath(path)) if path else None
        recs = score(rows, llm.chat(msgs, sp, lora_request=lora, use_tqdm=False), step=0)
        for x in recs:
            x["model"] = name
        m = {"model": name, "adapter": path or "", **{k: v for k, v in metrics(recs).items() if k != "step"}}
        all_recs += recs
        all_m.append(m)

    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_m[0]))
        w.writeheader()
        w.writerows(all_m)
    cols = ["model", "id", "gt", "read", "err_min", "score", "exact", "hour_ok", "think_ok", "thinking"]
    with open(f"{args.out}/per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_recs)
    with open(f"{args.out}/per_image.jsonl", "w") as f:
        for x in all_recs:
            f.write(json.dumps(x) + "\n")

    hdr = "| model | exact (±1 min) | within 5 min | hour correct | median err (min) | reward | think chars |"
    table = [hdr, "|---|---:|---:|---:|---:|---:|---:|"]
    for m in all_m:
        table.append(f"| {m['model']} | {m['exact']:.0%} | {m['within5']:.0%} | {m['hour_correct']:.0%} | "
                     f"{m['median_err_min']:.0f} | {m['reward']:.3f} | {m['think_chars']:.0f} |")
    names = [n for n, _ in runs]
    by = {(x["model"], x["id"]): x for x in all_recs}
    lines = [f"# Clock model comparison ({args.split}, {len(rows)} images, greedy)", "", *table, "",
             "| image | GT | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for r in rows:
        cells = [(f"**{by[(n, r['id'])]['read']}**" if by[(n, r["id"])]["exact"] else (by[(n, r["id"])]["read"] or "?"))
                 for n in names]
        lines.append(f"| {r['id']} | {r['time']} | " + " | ".join(cells) + " |")
    lines += ["", "Bold = exact (±1 min)."]
    with open(f"{args.out}/report.md", "w") as f:
        f.write("\n".join(lines))

    print("\n" + "\n".join(table))
    print(f"\n-> {args.out}/ (metrics.csv, per_image.csv, per_image.jsonl, report.md)")


if __name__ == "__main__":
    main()
