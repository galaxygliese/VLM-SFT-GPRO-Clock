"""Shared pieces for bbox GRPO: classes, prompt, output parsing, matching, metrics, rewards.

Coordinates: the model reads/writes Qwen3-VL's native relative format, integers in [0, 1000]
(x / width * 1000). Labels in data/fisheye8k/labels/*.jsonl are absolute pixels and are
converted when the dataset is built.
"""
import json
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

# traffic_light and road_symbol are intentionally excluded from the task
CLASSES = ["car", "bus_truck", "human", "bike", "traffic_sign", "crosswalk"]
CLASS_DESC = {
    "car": "passenger cars, vans, taxis, SUVs",
    "bus_truck": "buses and trucks",
    "human": "pedestrians and riders",
    "bike": "motorcycles, scooters and bicycles",
    "traffic_sign": "traffic signs",
    "crosswalk": "zebra crosswalk areas",
}
IOU_THR = 0.5

PROMPT = (
    "This is a fisheye traffic-camera image: straight lines bend and objects near the rim are small "
    "and distorted. Detect every object of these classes:\n"
    + "\n".join(f"- {c}: {CLASS_DESC[c]}" for c in CLASSES)
    + "\nReturn only a JSON list, one entry per object: "
    '[{"bbox_2d": [x1, y1, x2, y2], "label": "<class>"}, ...] '
    "with coordinates in the 0-1000 relative range. Return [] if there are no objects."
)

# Thinking mode: reason about labels and locations first, then answer. Only the JSON after
# </thinking> is scored, so boxes written inside the thinking earn nothing.
PROMPT_THINK = (
    "This is a fisheye traffic-camera image: straight lines bend and objects near the rim are small "
    "and distorted. Detect every object of these classes:\n"
    + "\n".join(f"- {c}: {CLASS_DESC[c]}" for c in CLASSES)
    + "\nFirst think inside <thinking></thinking>: scan the image region by region (top, left, center, "
    "right, bottom, and the rim), and for each region note which classes are present, how many, and where "
    "they are. Keep it brief. Then, after </thinking>, output only a JSON list, one entry per object: "
    '[{"bbox_2d": [x1, y1, x2, y2], "label": "<class>"}, ...] '
    "with coordinates in the 0-1000 relative range. Output [] if there are no objects.\n"
    "Format:\n<thinking>...</thinking>\n[...]"
)
THINK_RE = re.compile(r"^\s*<thinking>(.*?)</thinking>\s*(.*)$", re.S)


def get_prompt(thinking=False):
    return PROMPT_THINK if thinking else PROMPT


def split_thinking(text):
    """-> (thinking, answer, well_formed). well_formed: one leading, non-empty <thinking> block.

    No tag at all -> the whole text is the answer (non-thinking mode). An unclosed <thinking>
    (e.g. truncated) -> empty answer, so nothing is scored.
    """
    m = THINK_RE.match(text)
    if m:
        thinking, answer = m.group(1), m.group(2)
        ok = bool(thinking.strip()) and "<thinking>" not in thinking and "<thinking>" not in answer \
            and "</thinking>" not in answer
        return thinking.strip(), answer, ok
    if "<thinking>" in text or "</thinking>" in text:
        return "", "", False
    return "", text, False


# ---------------------------------------------------------------- data
def load_split(path, classes=CLASSES):
    """Read a labels jsonl -> list of {"image": path, "objects": [{"label", "bbox" (0-1000)}]}."""
    rows = []
    for line in open(path):
        r = json.loads(line)
        w, h = r["width"], r["height"]
        objs = []
        for o in r["objects"]:
            if o["label"] not in classes:
                continue
            x1, y1, x2, y2 = o["bbox"]
            objs.append({"label": o["label"],
                         "bbox": [round(x1 / w * 1000), round(y1 / h * 1000),
                                  round(x2 / w * 1000), round(y2 / h * 1000)]})
        rows.append({"image": r["image"], "objects": objs})
    return rows


def format_answer(objs):
    """Ground truth rendered in the expected output format (for length estimates / SFT)."""
    return json.dumps([{"bbox_2d": o["bbox"], "label": o["label"]} for o in objs])


# ---------------------------------------------------------------- parsing
_OBJ_RE = re.compile(
    r'\{\s*"bbox_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,'
    r'\s*(-?\d+(?:\.\d+)?)\s*\]\s*,\s*"label"\s*:\s*"([^"]*)"\s*\}')


def strip_fence(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)(?:```|$)", text, re.S)
    return m.group(1).strip() if m else text


def parse_objects(text):
    """Lenient parse: every well-formed {"bbox_2d": [...], "label": "..."} in the text.

    Works on truncated output too, so a long list cut by max length still gets credit for
    the objects it did finish. Returns (objects, strict_ok) where strict_ok means the whole
    output is a valid JSON list of valid objects.
    """
    body = strip_fence(text)
    objs = []
    for m in _OBJ_RE.finditer(body):
        x1, y1, x2, y2 = (float(v) for v in m.groups()[:4])
        objs.append({"label": m.group(5).strip(), "bbox": [x1, y1, x2, y2]})
    strict_ok = False
    try:
        data = json.loads(body)
        strict_ok = isinstance(data, list) and all(
            isinstance(d, dict) and set(d) == {"bbox_2d", "label"}
            and isinstance(d["bbox_2d"], list) and len(d["bbox_2d"]) == 4
            and all(isinstance(v, (int, float)) for v in d["bbox_2d"])
            for d in data) and len(data) == len(objs)
    except (json.JSONDecodeError, TypeError):
        pass
    return objs, strict_ok


def valid_box(b):
    x1, y1, x2, y2 = b
    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


# ---------------------------------------------------------------- matching
def iou_matrix(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = np.clip(a[:, 2:] - a[:, :2], 0, None).prod(-1)
    area_b = np.clip(b[:, 2:] - b[:, :2], 0, None).prod(-1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match(pred, gt, classes=CLASSES, iou_thr=IOU_THR):
    """Per-class Hungarian matching. Returns {class: (tp, n_pred, n_gt, sum_iou_of_tp)}."""
    out = {}
    for c in classes:
        p = [o["bbox"] for o in pred if o["label"] == c]
        g = [o["bbox"] for o in gt if o["label"] == c]
        tp, s = 0, 0.0
        if p and g:
            m = iou_matrix(p, g)
            r, k = linear_sum_assignment(-m)
            hit = m[r, k] >= iou_thr
            tp, s = int(hit.sum()), float(m[r, k][hit].sum())
        out[c] = (tp, len(p), len(g), s)
    return out


def prf(tp, n_pred, n_gt):
    p = tp / n_pred if n_pred else (1.0 if n_gt == 0 else 0.0)
    r = tp / n_gt if n_gt else (1.0 if n_pred == 0 else 0.0)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# ---------------------------------------------------------------- rewards
def detection_score(pred, gt):
    """Soft F1 in [0, 1]: 2 * sum(IoU of matched pairs with IoU >= 0.5) / (n_pred + n_gt).

    Equals F1@0.5 when every match is perfect, and also rewards tighter boxes.
    Predictions with unknown labels or invalid boxes count as false positives.
    """
    stats = match(pred, gt)
    s = sum(v[3] for v in stats.values())
    n = len(pred) + len(gt)
    return 1.0 if n == 0 else 2 * s / n


def completion_text(c):
    return c[0]["content"] if isinstance(c, list) else c


def answer_text(c):
    """The part of a completion that is scored (text after </thinking>, or all of it)."""
    return split_thinking(completion_text(c))[1]


def format_reward(completions, **kwargs):
    """1.0 for a clean JSON list with valid labels/boxes, partial credit otherwise."""
    rewards = []
    for c in completions:
        objs, ok = parse_objects(answer_text(c))
        if not ok:
            rewards.append(0.0)
            continue
        if not objs:
            rewards.append(1.0)
            continue
        good = sum(o["label"] in CLASSES and valid_box(o["bbox"]) for o in objs)
        rewards.append(good / len(objs))
    return rewards


def detection_reward(completions, objects, log_metric=None, **kwargs):
    """Soft-F1 detection reward against the pseudo-labels (`objects` column, JSON string)."""
    rewards, prec, rec = [], [], []
    for c, gt_json in zip(completions, objects):
        gt = json.loads(gt_json)
        pred, _ = parse_objects(answer_text(c))
        rewards.append(detection_score(pred, gt))
        st = match(pred, gt)
        tp = sum(v[0] for v in st.values())
        p, r, _ = prf(tp, len(pred), len(gt))
        prec.append(p)
        rec.append(r)
    if log_metric is not None:
        log_metric("det/precision@0.5", float(np.mean(prec)))
        log_metric("det/recall@0.5", float(np.mean(rec)))
    return rewards


THINK_MAX_CHARS = 2000  # soft cap: longer thinking gets linearly less credit (0 at 2x)


def thinking_format_reward(completions, log_metric=None, **kwargs):
    """1.0 for `<thinking>non-empty</thinking>` followed by the answer, scaled down past THINK_MAX_CHARS."""
    rewards, lens, oks = [], [], []
    for c in completions:
        thinking, _, ok = split_thinking(completion_text(c))
        r = 0.0
        if ok:
            over = max(0, len(thinking) - THINK_MAX_CHARS) / THINK_MAX_CHARS
            r = max(0.0, 1.0 - over)
        rewards.append(r)
        oks.append(ok)
        lens.append(len(thinking))
    if log_metric is not None:
        log_metric("think/well_formed", float(np.mean(oks)))
        log_metric("think/chars", float(np.mean(lens)))
    return rewards


def probe_score(text, probe):
    """For grpo_common.ThinkingProbeCallback: probe = {"image_path", "id", "objects" (0-1000 GT)}."""
    thinking, answer, ok = split_thinking(text)
    pred, strict = parse_objects(answer)
    st = match(pred, probe["objects"])
    tp = sum(v[0] for v in st.values())
    p, r, f = prf(tp, len(pred), len(probe["objects"]))
    return {"score": detection_score(pred, probe["objects"]), "P": p, "R": r, "F1": f,
            "n_pred": len(pred), "n_gt": len(probe["objects"]), "strict_json": strict,
            "thinking": thinking, "think_ok": ok,
            "answer": f"{len(pred)} boxes: " + ", ".join(f"{k}={sum(o['label'] == k for o in pred)}" for k in CLASSES)}
