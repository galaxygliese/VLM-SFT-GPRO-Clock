"""Synthetic analog clocks with exact labels, for the SFT warm-up (train_clock_sft.py).

Times are uniform over the 12-hour dial (no 10:10 bias). Faces vary in size, position, colors,
numerals (Arabic / Roman / 12-3-6-9 only / none), tick marks, hand shapes, an optional second hand,
rotation, aspect squash, blur and JPEG quality. The background is a single random color.

    python clock_synth.py --n 5000   # -> data/clock_synth/train/*.jpg + labels/train.jsonl

Labels use the data/clock schema ({"image", "hour", "minute", "time", ...}) so clock_utils.load_split works.
"""
import argparse
import glob
import json
import math
import os
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_GLOBS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans*.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif*.ttf",
              "/usr/share/fonts/truetype/liberation/Liberation*-Regular.ttf",
              "/usr/share/fonts/truetype/liberation/Liberation*-Bold.ttf",
              "/usr/share/fonts/truetype/noto/NotoSans-*.ttf", "/usr/share/fonts/truetype/noto/NotoSerif-*.ttf",
              "/usr/share/fonts/truetype/ubuntu/Ubuntu-[BRMC].ttf"]
FONTS = sorted({f for g in FONT_GLOBS for f in glob.glob(g) if "Mono" not in f and "Italic" not in f})
ROMAN = ["XII", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
SS = 2  # supersampling for anti-aliasing


def _rand_color(rng, light=None):
    if light is True:
        return tuple(rng.randint(200, 255) for _ in range(3))
    if light is False:
        return tuple(rng.randint(0, 70) for _ in range(3))
    return tuple(rng.randint(0, 255) for _ in range(3))


def _polar(cx, cy, r, deg):
    a = math.radians(deg - 90)  # 0 deg = 12 o'clock, clockwise
    return cx + r * math.cos(a), cy + r * math.sin(a)


def _hand(d, cx, cy, deg, length, width, color, style, tail=0.0):
    tip = _polar(cx, cy, length, deg)
    back = _polar(cx, cy, -tail * length, deg)
    if style == "line":
        d.line([back, tip], fill=color, width=max(1, int(width)))
    else:  # tapered / spade polygon
        l1 = _polar(cx, cy, width / 2, deg - 90)
        r1 = _polar(cx, cy, width / 2, deg + 90)
        if style == "taper":
            d.polygon([back, l1, tip, r1], fill=color)
        else:  # "arrow": wide body then a point
            mid = _polar(cx, cy, length * 0.8, deg)
            ml, mr = _polar(mid[0], mid[1], width * 0.9, deg - 90), _polar(mid[0], mid[1], width * 0.9, deg + 90)
            d.line([back, mid], fill=color, width=max(1, int(width * 0.7)))
            d.polygon([ml, tip, mr], fill=color)


def render_clock(rng, hour, minute):
    """-> PIL RGB image of an analog clock showing hour:minute (hour 1-12)."""
    W, H = rng.choice([(512, 512), (640, 480), (480, 640), (768, 512), (448, 448)])
    canvas = Image.new("RGB", (W, H), _rand_color(rng))

    R = int(min(W, H) * rng.uniform(0.12, 0.46))  # clock radius on the canvas
    S = 2 * R * SS + 8
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    c, r = S / 2, R * SS

    dark_face = rng.random() < 0.2
    face = _rand_color(rng, light=not dark_face)
    ink = _rand_color(rng, light=dark_face)
    square = rng.random() < 0.12
    rim_w = int(r * rng.uniform(0.0, 0.12))
    rim_col = _rand_color(rng)
    shape = d.rounded_rectangle if square else d.ellipse
    kw = {"radius": int(r * 0.2)} if square else {}
    shape([c - r, c - r, c + r, c + r], fill=rim_col if rim_w else face, **kw)
    ri = r - rim_w
    shape([c - ri, c - ri, c + ri, c + ri], fill=face, **({"radius": int(ri * 0.2)} if square else {}))

    # ticks
    tick_mode = rng.choice(["all", "all", "hours", "none"])
    for i in range(60):
        if tick_mode == "none" or (tick_mode == "hours" and i % 5):
            continue
        big = i % 5 == 0
        L = ri * (0.12 if big else 0.05)
        p0, p1 = _polar(c, c, ri * 0.95, i * 6), _polar(c, c, ri * 0.95 - L, i * 6)
        d.line([p0, p1], fill=ink, width=max(1, int(ri * (0.025 if big else 0.01))))

    # numerals
    num_mode = rng.choices(["arabic", "roman", "quarters", "none"], weights=[5, 2, 1.5, 1.5])[0]
    if num_mode != "none":
        font_path = rng.choice(FONTS) if FONTS else None
        fs = int(ri * rng.uniform(0.16, 0.26) * (0.8 if num_mode == "roman" else 1.0))
        font = ImageFont.truetype(font_path, fs) if font_path else ImageFont.load_default(size=fs)
        nr = ri * (0.72 if tick_mode != "none" else 0.82)
        for i in range(12):
            if num_mode == "quarters" and i % 3:
                continue
            txt = ROMAN[i] if num_mode == "roman" else str(12 if i == 0 else i)
            x, y = _polar(c, c, nr, i * 30)
            d.text((x, y), txt, fill=ink, font=font, anchor="mm")

    # hands (angles follow the label exactly)
    h_deg = ((hour % 12) + minute / 60) * 30
    m_deg = minute * 6
    style = rng.choice(["line", "taper", "arrow"])
    hand_col = ink if rng.random() < 0.8 else _rand_color(rng, light=dark_face)
    tail = rng.choice([0.0, 0.0, 0.15])
    _hand(d, c, c, h_deg, ri * rng.uniform(0.45, 0.58), ri * rng.uniform(0.05, 0.09), hand_col, style, tail)
    _hand(d, c, c, m_deg, ri * rng.uniform(0.75, 0.9), ri * rng.uniform(0.03, 0.055), hand_col, style, tail)
    if rng.random() < 0.4:  # second hand: thin, long, often red, random angle
        s_col = (200, 30, 30) if rng.random() < 0.7 else hand_col
        _hand(d, c, c, rng.uniform(0, 360), ri * 0.92, max(1, ri * 0.012), s_col, "line", 0.2)
    cap = ri * rng.uniform(0.03, 0.07)
    d.ellipse([c - cap, c - cap, c + cap, c + cap], fill=hand_col)

    # geometric jitter: small rotation (the whole face, so the reading is unchanged) + perspective-like squash
    layer = layer.rotate(rng.uniform(-15, 15), resample=Image.BICUBIC, expand=True)
    sx, sy = (rng.uniform(0.8, 1.0), 1.0) if rng.random() < 0.5 else (1.0, rng.uniform(0.8, 1.0))
    if rng.random() < 0.6:
        sx = sy = 1.0
    layer = layer.resize((max(8, int(layer.width * sx / SS)), max(8, int(layer.height * sy / SS))), Image.LANCZOS)

    x = rng.randint(0, max(0, W - layer.width))
    y = rng.randint(0, max(0, H - layer.height))
    canvas.paste(layer, (x, y), layer)
    if rng.random() < 0.3:
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/clock_synth")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(f"{args.out}/{args.split}", exist_ok=True)
    os.makedirs(f"{args.out}/labels", exist_ok=True)
    with open(f"{args.out}/labels/{args.split}.jsonl", "w") as f:
        for i in range(args.n):
            t = rng.randrange(720)
            hour, minute = (t // 60) or 12, t % 60
            img = render_clock(rng, hour, minute)
            rel = f"{args.split}/synth_{i:05d}.jpg"
            img.save(f"{args.out}/{rel}", quality=rng.randint(60, 95))
            f.write(json.dumps({"image": rel, "width": img.width, "height": img.height, "hour": hour,
                                "minute": minute, "time": f"{hour:02d}:{minute:02d}", "source": "synthetic"}) + "\n")
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{args.n}")
    print(f"{args.n} images -> {args.out}/{args.split}/, labels -> {args.out}/labels/{args.split}.jsonl")


if __name__ == "__main__":
    main()
