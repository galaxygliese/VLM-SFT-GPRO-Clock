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


def _num(i):
    return 12 if i % 12 == 0 else i % 12


def think_template(hour, minute, rng=None):
    """Reference reasoning for SFT, derived from the label: the steps PROMPT_THINK asks for.

    The hour-hand position follows from the minutes (hour + minute/60), so the text is faithful to
    any correctly labelled clock. `rng` (random.Random) varies the wording a little.
    """
    h, m = _num(hour), minute
    nxt = _num(h + 1)
    f = m / 60
    if m == 0:
        hour_pos = f"pointing at the {h}"
    elif f < 0.15:
        hour_pos = f"just past the {h}"
    elif f < 0.35:
        hour_pos = f"about a quarter of the way from the {h} to the {nxt}"
    elif f < 0.6:
        hour_pos = f"about halfway between the {h} and the {nxt}"
    elif f < 0.85:
        hour_pos = f"about three quarters of the way from the {h} to the {nxt}"
    else:
        hour_pos = f"almost at the {nxt} but not past it"

    n, k = m // 5, m % 5
    mark = _num(n)
    if k == 0:
        minute_pos = f"points at the {mark}" + (", so 00 minutes" if m == 0 else f", so {m} minutes")
    else:
        ticks = "tick" if k == 1 else "ticks"
        base = 0 if n == 0 else n * 5
        minute_pos = f"is {k} {ticks} past the {mark}, so {base} + {k} = {m} minutes"

    if m == 0:
        check = f"With 00 minutes the hour hand should sit on the {h}, and it does."
    elif m >= 45:
        check = (f"At {m} minutes the hour hand should be close to the {nxt} without reaching it, "
                 f"which matches, so the hour is still {h}.")
    else:
        check = f"At {m} minutes the hour hand should be {int(round(f * 100))}% of the way past the {h}, which matches."

    opener = "The clock face is visible."
    if rng is not None:
        opener = rng.choice(["The clock face is visible.", "I can see the analog clock face.",
                             "Locating the clock face.", "There is one analog clock."])
    return (f"{opener} The hour hand (shorter) is {hour_pos}. The minute hand (longer) {minute_pos}. "
            f"{check} The time is {h}:{m:02d}.")


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
