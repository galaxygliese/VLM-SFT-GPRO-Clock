"""Auto-label fisheye images with bounding boxes using SAM3 (text-prompted, open-vocabulary).

The boxes are *pseudo-labels*: review them (see --vis) before trusting them as ground truth.

Output (one file per split):
    data/fisheye8k/labels/<split>.jsonl
        {"image": "train/xxx.png", "width": 1080, "height": 1080,
         "objects": [{"label": "car", "bbox": [x1, y1, x2, y2], "score": 0.93}, ...]}
    bbox is absolute pixel xyxy (ints).

Usage:
    python tools/make_labels.py                     # all splits
    python tools/make_labels.py --split eval --vis  # also draw boxes to data/fisheye8k/vis/<split>/
"""
import argparse
import glob
import json
import os
from collections import Counter

import torch
from PIL import Image, ImageDraw
from torchvision.ops import box_iou, nms

# class -> (SAM3 text prompts, confidence threshold)
CLASSES = {
    "car": (["car"], 0.40),
    "bus_truck": (["bus", "truck"], 0.40),
    "human": (["person"], 0.40),
    "bike": (["motorcycle", "scooter", "bicycle"], 0.40),
    "traffic_light": (["traffic light"], 0.40),
    "traffic_sign": (["traffic sign"], 0.45),
    "road_symbol": (["bicycle symbol on road", "arrow marking on road", "text painted on road"], 0.50),
    "crosswalk": (["crosswalk"], 0.60),
}
COLORS = {
    "car": "#3b82f6", "bus_truck": "#f59e0b", "human": "#10b981", "bike": "#ec4899",
    "traffic_light": "#eab308", "traffic_sign": "#ef4444", "road_symbol": "#a855f7", "crosswalk": "#06b6d4",
}
MOVING = ["car", "bus_truck", "human", "bike"]
MIN_SIDE = 6  # px; drop slivers


def default_ckpt():
    hits = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt"))
    if not hits:
        raise FileNotFoundError("sam3.pt not found; run `hf download facebook/sam3`")
    return hits[0]


def ioa(a, b):
    """Intersection over area of each box in a, vs every box in b -> [len(a), len(b)]."""
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1).clamp(min=1e-6)
    return inter / area_a[:, None]


def postprocess(boxes, scores, labels, w, h):
    """boxes [N,4] xyxy, scores [N], labels list[str] -> list of object dicts."""
    if len(boxes) == 0:
        return []
    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
    wh = boxes[:, 2:] - boxes[:, :2]
    keep = (wh >= MIN_SIDE).all(1)
    thr = torch.tensor([CLASSES[l][1] for l in labels])
    keep &= scores >= thr
    idx = keep.nonzero().flatten().tolist()
    boxes, scores, labels = boxes[idx], scores[idx], [labels[i] for i in idx]

    # 1) per-class NMS (merges duplicate prompts, e.g. motorcycle vs scooter)
    out = []
    for c in CLASSES:
        ci = [i for i, l in enumerate(labels) if l == c]
        if not ci:
            continue
        k = nms(boxes[ci], scores[ci], 0.5)
        out += [ci[j] for j in k.tolist()]
    boxes, scores, labels = boxes[out], scores[out], [labels[i] for i in out]

    # 2) car vs bus_truck: same object -> keep the higher score
    drop = set()
    for a in ("car", "bus_truck"):
        b = "bus_truck" if a == "car" else "car"
        ia = [i for i, l in enumerate(labels) if l == a]
        ib = [i for i, l in enumerate(labels) if l == b]
        if ia and ib:
            iou = box_iou(boxes[ia], boxes[ib])
            for r, i in enumerate(ia):
                for s, j in enumerate(ib):
                    if iou[r, s] > 0.6 and scores[i] < scores[j]:
                        drop.add(i)

    # 3) road_symbol false positives: on vehicles/people, or a crosswalk stripe
    isym = [i for i, l in enumerate(labels) if l == "road_symbol"]
    imov = [i for i, l in enumerate(labels) if l in MOVING]
    icw = [i for i, l in enumerate(labels) if l == "crosswalk"]
    if isym and imov:
        hit = (ioa(boxes[isym], boxes[imov]) > 0.3).any(1)
        drop |= {isym[r] for r in hit.nonzero().flatten().tolist()}
    if isym and icw:
        hit = (ioa(boxes[isym], boxes[icw]) > 0.5).any(1)
        drop |= {isym[r] for r in hit.nonzero().flatten().tolist()}
    # road_symbol bigger than 1/4 of the image is a whole road region, not a symbol
    for i in isym:
        bw, bh = (boxes[i, 2:] - boxes[i, :2]).tolist()
        if bw * bh > 0.25 * w * h:
            drop.add(i)

    objs = []
    for i in range(len(labels)):
        if i in drop:
            continue
        objs.append({"label": labels[i],
                     "bbox": [int(round(v)) for v in boxes[i].tolist()],
                     "score": round(float(scores[i]), 3)})
    objs.sort(key=lambda o: (list(CLASSES).index(o["label"]), o["bbox"][1], o["bbox"][0]))
    return objs


def draw(img_path, objs, out_path):
    im = Image.open(img_path).convert("RGB")
    d = ImageDraw.Draw(im)
    for o in objs:
        c = COLORS[o["label"]]
        d.rectangle(o["bbox"], outline=c, width=3)
        x1, y1 = o["bbox"][:2]
        txt = f'{o["label"]} {o["score"]:.2f}'
        tw = d.textlength(txt)
        d.rectangle([x1, max(0, y1 - 12), x1 + tw + 4, max(12, y1)], fill=c)
        d.text((x1 + 2, max(0, y1 - 12)), txt, fill="white")
    im.save(out_path, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/fisheye8k")
    ap.add_argument("--split", nargs="+", default=["train", "eval"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--vis", action="store_true", help="save box overlays to <root>/vis/<split>/")
    args = ap.parse_args()

    from ultralytics.models.sam import SAM3SemanticPredictor

    pred = SAM3SemanticPredictor(overrides=dict(
        conf=min(t for _, t in CLASSES.values()), task="segment", mode="predict",
        model=args.ckpt or default_ckpt(), half=True, save=False, verbose=False))
    prompts, prompt2cls = [], []
    for c, (ps, _) in CLASSES.items():
        prompts += ps
        prompt2cls += [c] * len(ps)

    os.makedirs(f"{args.root}/labels", exist_ok=True)
    for split in args.split:
        paths = sorted(glob.glob(f"{args.root}/{split}/*.png") + glob.glob(f"{args.root}/{split}/*.jpg"))
        if args.vis:
            os.makedirs(f"{args.root}/vis/{split}", exist_ok=True)
        stats = Counter()
        out_file = f"{args.root}/labels/{split}.jsonl"
        with open(out_file, "w") as f:
            for n, p in enumerate(paths):
                w, h = Image.open(p).size
                pred.set_image(p)
                r = pred(text=prompts)[0]
                b = r.boxes
                objs = postprocess(b.xyxy.float().cpu(), b.conf.float().cpu(),
                                   [prompt2cls[int(c)] for c in b.cls.tolist()], w, h)
                stats.update(o["label"] for o in objs)
                rel = os.path.relpath(p, args.root)
                f.write(json.dumps({"image": rel, "width": w, "height": h, "objects": objs}) + "\n")
                if args.vis:
                    draw(p, objs, f"{args.root}/vis/{split}/{os.path.splitext(os.path.basename(p))[0]}.jpg")
                print(f"[{split}] {n + 1}/{len(paths)} {rel}: {len(objs)} objects", flush=True)
        print(f"\n{split}: {len(paths)} images -> {out_file}")
        for c in CLASSES:
            print(f"  {c:14s} {stats[c]:5d}")


if __name__ == "__main__":
    main()
