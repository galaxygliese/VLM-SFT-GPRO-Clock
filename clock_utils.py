"""Analog-clock reading task (TickTockVQA subset, data/clock): prompt, parsing, rewards, metrics.

Answer: 12-hour time "HH:MM" after </thinking>. Error is measured on the 12-hour dial in minutes,
wrapping around (11:58 vs 12:02 = 4 min).
"""
import json
import re

import numpy as np

from bbox_utils import split_thinking, thinking_format_reward  # noqa: F401  (re-exported)

PROMPT = ("What time does the analog clock in this image show? "
          "Answer with only the 12-hour time as HH:MM.")

PROMPT_THINK = (
    "What time does the analog clock in this image show?\n"
    "First think inside <thinking></thinking>: find the clock face, identify the hour hand (shorter, "
    "thicker) and the minute hand (longer, thinner), say which number or tick each hand points at, "
    "work out the minutes from the minute hand, check that the hour hand's position between two numbers "
    "agrees with those minutes, then state the time. Keep it brief.\n"
    "Then, after </thinking>, output only the 12-hour time as HH:MM.\n"
    "Format:\n<thinking>...</thinking>\nHH:MM"
)

TIME_RE = re.compile(r"(\d{1,2})\s*:\s*(\d{2})")
STRICT_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")
FULL_CREDIT_MIN = 1   # label noise: +-1 min counts as exact
ZERO_CREDIT_MIN = 30  # linear partial credit down to 0 at 30 min off


def get_prompt(thinking=False):
    return PROMPT_THINK if thinking else PROMPT


def load_split(path, root="data/clock"):
    rows = []
    for line in open(path):
        r = json.loads(line)
        rows.append({"image_path": f"{root}/{r['image']}", "id": r["image"], "time": r["time"],
                     "hour": r["hour"], "minute": r["minute"]})
    return rows


def answer_of(text):
    """Text that is scored: after </thinking> (or all of it when there is no thinking tag)."""
    return split_thinking(text)[1]


def parse_time(answer):
    """-> (minutes on a 12h dial in [0, 720), strict) or (None, False)."""
    m = TIME_RE.search(answer)
    if not m:
        return None, False
    h, mi = int(m.group(1)), int(m.group(2))
    if not (0 <= h <= 12 and 0 <= mi <= 59):
        return None, False
    return (h % 12) * 60 + mi, bool(STRICT_RE.match(answer))


def dial_error(pred_min, hour, minute):
    gt = (hour % 12) * 60 + minute
    d = abs(pred_min - gt) % 720
    return min(d, 720 - d)


def time_score(text, hour, minute):
    pred, _ = parse_time(answer_of(text))
    if pred is None:
        return 0.0, None
    err = dial_error(pred, hour, minute)
    if err <= FULL_CREDIT_MIN:
        return 1.0, err
    return max(0.0, 1.0 - (err - FULL_CREDIT_MIN) / (ZERO_CREDIT_MIN - FULL_CREDIT_MIN)), err


def _text(c):
    return c[0]["content"] if isinstance(c, list) else c


# ---------------------------------------------------------------- GRPO rewards
def time_reward(completions, hour, minute, log_metric=None, **kwargs):
    """1.0 within 1 min on the 12h dial, linear down to 0 at 30 min off; unparseable -> 0."""
    rewards, errs = [], []
    for c, h, m in zip(completions, hour, minute):
        r, err = time_score(_text(c), h, m)
        rewards.append(r)
        errs.append(err)
    if log_metric is not None:
        ok = [e for e in errs if e is not None]
        log_metric("clock/exact", float(np.mean([e is not None and e <= FULL_CREDIT_MIN for e in errs])))
        log_metric("clock/within5", float(np.mean([e is not None and e <= 5 for e in errs])))
        log_metric("clock/parsed", len(ok) / len(errs))
        if ok:
            log_metric("clock/median_err_min", float(np.median(ok)))
    return rewards


def format_reward(completions, **kwargs):
    """1.0 if the answer part is exactly HH:MM."""
    return [1.0 if parse_time(answer_of(_text(c)))[1] else 0.0 for c in completions]


def probe_score(text, probe):
    """For ThinkingProbeCallback."""
    thinking, answer, ok = split_thinking(text)
    score, err = time_score(text, probe["hour"], probe["minute"])
    return {"score": score, "err_min": err, "gt": probe["time"], "answer": answer.strip()[:50],
            "thinking": thinking, "think_ok": ok}
