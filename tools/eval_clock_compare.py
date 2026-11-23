"""Compare clock-reading models on the eval split in one vLLM load: the base model + any LoRA adapters.

    python tools/eval_clock_compare.py \\
        sft=outputs/sft-clock-think/final \\
        grpo=outputs/grpo-clock-think-mix/final \\
        sft_grpo=outputs/grpo-clock-think-sft/final

Each positional argument is NAME=ADAPTER_DIR ("base" is always evaluated first unless --no_base).
Greedy decoding, thinking prompt, same scoring as tools/eval_clock_checkpoints.py.

Also reports the TickTockVQA paper's metrics (arXiv 2603.08011, Table 1), so results on the full test set
(tools/prepare_clock_testset.py) compare directly with the paper and with It's About Time:
    hour_acc   predicted hour == GT hour (12-hour dial)
    min_acc    minute within +-MIN_TOL (default 2, the paper's tolerance), on the 60-minute dial
    full_acc   hour_acc and min_acc ("B" in the paper)
    full_swap  full_acc, or the reading matches with hour and minute hands swapped ("S", e.g. 06:18 for 03:30)
    mae_min    mean dial error in minutes over parsed answers (12-hour wraparound); unparsed counted in parse_rate
plus full_acc per source (COCO / OpenImages / ClockMovies) when the labels carry a "source" field.

    python tools/eval_clock_compare.py --root data/clock_full --split test --out outputs/bench_ticktock \
        sft=outputs/sft-clock-think/final sft_grpo=outputs/grpo-clock-think-sft/final

Writes to --out (default outputs/compare_clock/):
    metrics.csv    one row per model: exact, within5, hour_correct, median_err_min, reward, think stats, paper metrics
    per_image.csv  one row per (model, image): gt, read time, error, thinking
    report.md      metrics tables (+ read time per image for every model when <= 200 images)
"""
import argparse
import csv
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clock_utils import TIME_RE, get_prompt, load_split  # noqa: E402
from eval_clock_checkpoints import metrics, score  # noqa: E402


def read_hm(text):
    m = TIME_RE.search(text or "")
    if not m or not (0 <= int(m.group(1)) <= 12 and 0 <= int(m.group(2)) <= 59):
        return None
    return int(m.group(1)), int(m.group(2))


def _min_diff(a, b):
    d = abs(a - b) % 60
    return min(d, 60 - d)


def paper_flags(read, hour, minute, tol):
    """-> (hour_ok, min_ok, full_ok, full_or_swap_ok) as in the TickTockVQA paper."""
    p = read_hm(read)
    if p is None:
        return False, False, False, False
    ph, pm = p
    hour_ok = ph % 12 == hour % 12
    min_ok = _min_diff(pm, minute) <= tol
    full = hour_ok and min_ok
    # hands swapped: the minute hand read as hour (number it points at, lower one) and the hour hand read as minutes
    swap = ph % 12 == (minute // 5) % 12 and _min_diff(pm, (hour % 12) * 5 + minute / 12) <= tol
    return hour_ok, min_ok, full, full or swap


def paper_metrics(recs, rows, tol):
    flags = [paper_flags(x["read"], r["hour"], r["minute"], tol) for x, r in zip(recs, rows)]
    n = len(recs)
    errs = [x["err_min"] for x in recs if x["err_min"] is not None]
    out = {"hour_acc": sum(f[0] for f in flags) / n, "min_acc": sum(f[1] for f in flags) / n,
           "full_acc": sum(f[2] for f in flags) / n, "full_swap": sum(f[3] for f in flags) / n,
           "mae_min": sum(errs) / len(errs) if errs else float("nan"), "parse_rate": len(errs) / n}
    for src in sorted({r.get("source") for r in rows if r.get("source")}):
        idx = [i for i, r in enumerate(rows) if r.get("source") == src]
        out[f"full_acc_{src}"] = sum(flags[i][2] for i in idx) / len(idx)
    for x, f in zip(recs, flags):
        x["paper_full"], x["paper_swap"] = f[2], f[3]
    return out


def load_image(path, max_side):
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapters", nargs="*", help="NAME=ADAPTER_DIR")
    ap.add_argument("--model", default="checkpoints/Qwen3-VL-4B-Instruct-bf16")
    ap.add_argument("--root", default="data/clock")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--no_base", action="store_true")
    ap.add_argument("--out", default="outputs/compare_clock")
    ap.add_argument("--min_tol", type=int, default=2, help="minute tolerance for the paper metrics")
    ap.add_argument("--chunk", type=int, default=512, help="images decoded at a time (RAM)")
    ap.add_argument("--max_side", type=int, default=1920,
                    help="downscale images whose long side is larger (Clock Movies frames are up to 3840 px)")
    args = ap.parse_args()

    runs = [] if args.no_base else [("base", None)]
    for a in args.adapters:
        name, _, path = a.partition("=")
        if not path or not os.path.isfile(os.path.join(path, "adapter_config.json")):
            sys.exit(f"bad adapter '{a}': expected NAME=DIR with DIR/adapter_config.json")
        runs.append((name, path))

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    label_path = f"{args.root}/labels/{args.split}.jsonl"
    rows = load_split(label_path, args.root)
    for r, line in zip(rows, open(label_path)):
        r["source"] = json.loads(line).get("source")
    # max_model_len: room for up to ~3.6k image tokens (1920x1920) + prompt + completion
    llm = LLM(model=args.model, max_model_len=args.max_tokens + 4096, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"image": 1}, enable_lora=True, max_lora_rank=128, max_loras=1,
              max_num_seqs=32)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    loras = [LoRARequest(name, i + 1, os.path.abspath(path)) if path else None for i, (name, path) in enumerate(runs)]
    outs = {name: [] for name, _ in runs}
    for c in range(0, len(rows), args.chunk):  # decode images chunk by chunk; every model sees each chunk
        chunk = rows[c:c + args.chunk]
        msgs = [[{"role": "user", "content": [
            {"type": "image_pil", "image_pil": load_image(r["image_path"], args.max_side)},
            {"type": "text", "text": get_prompt(True)}]}] for r in chunk]
        for (name, _), lora in zip(runs, loras):
            outs[name] += llm.chat(msgs, sp, lora_request=lora, use_tqdm=False)
        print(f"  {min(c + args.chunk, len(rows))}/{len(rows)} images", flush=True)

    all_recs, all_m = [], []
    for name, path in runs:
        recs = score(rows, outs[name], step=0)
        for x in recs:
            x["model"] = name
        m = {"model": name, "adapter": path or "", **{k: v for k, v in metrics(recs).items() if k != "step"},
             **paper_metrics(recs, rows, args.min_tol)}
        all_recs += recs
        all_m.append(m)

    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_m[0]))
        w.writeheader()
        w.writerows(all_m)
    cols = ["model", "id", "gt", "read", "err_min", "score", "exact", "hour_ok", "paper_full", "paper_swap",
            "think_ok", "thinking"]
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
    srcs = [k for k in all_m[0] if k.startswith("full_acc_")]
    ptable = ["| model | hour acc | min acc (±%d) | full-time acc | full-time (swap ok) | MAE (min) | parsed |"
              % args.min_tol + "".join(f" {k[9:]} |" for k in srcs),
              "|---|---:|---:|---:|---:|---:|---:|" + "---:|" * len(srcs)]
    for m in all_m:
        ptable.append(f"| {m['model']} | {100 * m['hour_acc']:.2f} | {100 * m['min_acc']:.2f} | "
                      f"{100 * m['full_acc']:.2f} | {100 * m['full_swap']:.2f} | {m['mae_min']:.2f} | "
                      f"{m['parse_rate']:.0%} |" + "".join(f" {100 * m[k]:.2f} |" for k in srcs))
    names = [n for n, _ in runs]
    by = {(x["model"], x["id"]): x for x in all_recs}
    lines = [f"# Clock model comparison ({args.split}, {len(rows)} images, greedy)", "", *table, "",
             "TickTockVQA paper metrics:", "", *ptable, ""]
    if len(rows) <= 200:
        lines += ["| image | GT | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        for r in rows:
            cells = [(f"**{by[(n, r['id'])]['read']}**" if by[(n, r["id"])]["exact"]
                      else (by[(n, r["id"])]["read"] or "?")) for n in names]
            lines.append(f"| {r['id']} | {r['time']} | " + " | ".join(cells) + " |")
        lines += ["", "Bold = exact (±1 min)."]
    with open(f"{args.out}/report.md", "w") as f:
        f.write("\n".join(lines))

    print("\n" + "\n".join(table) + "\n\nTickTockVQA paper metrics:\n" + "\n".join(ptable))
    print(f"\n-> {args.out}/ (metrics.csv, per_image.csv, per_image.jsonl, report.md)")


if __name__ == "__main__":
    main()
