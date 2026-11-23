"""Export the FULL TickTockVQA test split (5,247 images: COCO 2,063 / Open Images 1,940 / Clock Movies 1,244)
for benchmarking against the TickTockVQA paper (arXiv 2603.08011, Table 1), which includes It's About Time.

Unlike tools/prepare_clock_data.py there is no filtering: every clock type, transformation and license is
kept. Clock Movies has no redistribution rights: use it for local evaluation only, never republish its images.

    python tools/prepare_clock_testset.py        # -> data/clock_full/test/*.jpg + data/clock_full/labels/test.jsonl
    python eval_clock.py --root data/clock_full --split test --thinking --adapter outputs/sft-clock-think/final

Images are written as the original JPEG bytes (no re-encoding). Labels use the data/clock schema
(hour 1-12, minute, "time" hh:mm) plus all metadata, so clock_utils.load_split works unchanged.
"""
import argparse
import glob
import json
import os

import pyarrow.parquet as pq

KEEP = ["ampm", "time_string", "clock_type", "design", "environment", "transformation", "source", "source_url",
        "license", "license_type", "redistribution_allowed", "commercial_use", "image_name", "image_hash"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/ticktockvqa", help="TickTockVQA snapshot (parquet shards)")
    ap.add_argument("--out", default="data/clock_full")
    args = ap.parse_args()

    shards = sorted(glob.glob(f"{args.src}/test-*.parquet"))
    if not shards:
        raise SystemExit(f"no test shards in {args.src}; download with "
                         "huggingface_hub.snapshot_download('Jaeha-Choi/TickTockVQA', repo_type='dataset')")
    os.makedirs(f"{args.out}/test", exist_ok=True)
    os.makedirs(f"{args.out}/labels", exist_ok=True)
    n, odd, counts = 0, [], {}
    with open(f"{args.out}/labels/test.jsonl", "w") as f:
        for shard in shards:
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=256):
                for r in batch.to_pylist():
                    h, m = int(r["hour"]), int(r["minute"])
                    if not (1 <= h <= 12):
                        odd.append(f"{r['image_name']}: hour={h}")
                    hour = (h % 12) or 12  # 0 -> 12, 15 -> 3 (one bad label); dial reading is 12-hour
                    rel = f"test/{r['image_name']}"
                    with open(f"{args.out}/{rel}", "wb") as img:
                        img.write(r["image"]["bytes"])
                    row = {"image": rel, "hour": hour, "minute": m, "time": f"{hour:02d}:{m:02d}",
                           **{k: (list(r[k]) if k == "design" and r[k] is not None else r[k]) for k in KEEP}}
                    f.write(json.dumps(row, default=str) + "\n")
                    counts[r["source"]] = counts.get(r["source"], 0) + 1
                    n += 1
    print(f"{n} test images -> {args.out}/test/, labels -> {args.out}/labels/test.jsonl")
    print("  by source:", counts)
    if odd:
        print(f"  {len(odd)} labels with hour outside 1-12 (mapped onto the 12-hour dial):", "; ".join(odd))


if __name__ == "__main__":
    main()
