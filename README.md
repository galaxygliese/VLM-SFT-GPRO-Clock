# VLM-GPRO-BBox

GRPO training of a VLM for bounding-box detection on fisheye traffic-camera images (RTX 5090, 32 GB).

Environment: `~/.dev_venv` (torch 2.13, transformers 5.19, TRL 1.17, peft 0.19, vLLM 0.32, ultralytics 8.4).

## 1. Data and labels

Images are in `data/fisheye8k/{train,eval}/` (133 train / 36 eval, 1080×1080 PNG). `data/` is git-ignored.

The images came without labels, so the bounding boxes are **pseudo-labels** from SAM3, an open-vocabulary detector driven by text prompts (`facebook/sam3`, loaded through ultralytics).

```bash
python tools/make_labels.py --vis   # writes data/fisheye8k/labels/{train,eval}.jsonl and data/fisheye8k/vis/
```

Each JSONL line has this format (the bbox is absolute pixel `xyxy`):

```json
{"image": "train/xxx.png", "width": 1080, "height": 1080,
 "objects": [{"label": "car", "bbox": [x1, y1, x2, y2], "score": 0.93}]}
```

| class | SAM3 prompts | conf | train | eval |
|---|---|---:|---:|---:|
| car | car | 0.40 | 690 | 320 |
| bus_truck | bus, truck | 0.40 | 173 | 45 |
| human | person | 0.40 | 702 | 306 |
| bike | motorcycle, scooter, bicycle | 0.40 | 1158 | 481 |
| traffic_light | traffic light | 0.40 | 153 | 26 |
| traffic_sign | traffic sign | 0.45 | 180 | 12 |
| road_symbol | bicycle symbol / arrow / text painted on road | 0.50 | 357 | 110 |
| crosswalk | crosswalk | 0.60 | 338 | 49 |

Post-processing:
- Prompts that map to the same class go through NMS together.
- A box detected as both `car` and `bus_truck` keeps only the higher-scoring class.
- A `road_symbol` box is dropped if it overlaps a vehicle, person or crosswalk, or if it covers more than a quarter of the image.

Known limitations:
- `road_symbol` is the noisiest class: it sometimes fires on manhole covers and misses some painted text.
- `crosswalk` boxes can overlap at corners.
- Riders are labelled as `human`, and their vehicle is labelled separately as `bike`.

Check the images in `vis/` and fix the eval split by hand before using it as a benchmark.

## 2. Model selection

Candidates come from `../VLM-study`. Training needs BF16 weights. The cached Qwen FP8 checkpoints are dequantized to BF16 at load time (`FineGrainedFP8Config(dequantize=True)`). For full fidelity, use the original BF16 repo instead (`hf download Qwen/Qwen3-VL-4B-Instruct`).

`tools/check_vram.py` measures one GRPO policy step. The test uses 1 image (max 1024² px, 1078 prompt tokens), G completions × 512 tokens, LoRA r=64 on all linear layers, and gradient checkpointing.

| model | BF16 weights | G=8 peak | G=4 peak | zero-shot bbox output |
|---|---:|---:|---:|---|
| **Qwen3-VL-4B-Instruct** | 9.0 GiB | **21.9 GiB** | – | yes, `bbox_2d` JSON in 0–1000 coords |
| Qwen3-VL-8B-Instruct | 16.3 GiB | OOM | 24.4 GiB | yes |

**Selected: Qwen3-VL-4B-Instruct + LoRA.**
- It fits G=8 with about 10 GB left for rollouts.
- It already produces grounding JSON zero-shot, so the reward signal is non-zero from step 0.
- In the VLM-study fisheye benchmark it matches InternVL3.5-4B, the best of the 4B models: 84% with the fisheye prompt, and 90% with zoom tiles.
- TRL's `GRPOTrainer` supports Qwen-VL natively.
- The 8B leaves no room for generation, even at G=4.
- Full fine-tuning of the 4B would need about 64 GB of optimizer state, so it is out.
- InternVL3.5 and Molmo2 do not have Qwen's native `bbox_2d` grounding format.

```bash
python tools/check_vram.py --model Qwen/Qwen3-VL-4B-Instruct-FP8 --G 8 --L 512
```

## 3. GRPO training (object detection)

The task covers 6 classes: `car`, `bus_truck`, `human`, `bike`, `traffic_sign` and `crosswalk`. `traffic_light` and `road_symbol` are left out. The model returns a JSON list of `{"bbox_2d": [x1, y1, x2, y2], "label": ...}`, with coordinates in Qwen's 0–1000 relative range. The prompt is in `bbox_utils.py`.

```bash
python tools/dequantize_fp8.py   # once: BF16 copy -> checkpoints/Qwen3-VL-4B-Instruct-bf16 (9.1 GB)
python eval_bbox.py              # baseline accuracy on the eval split
python train_bbox_gpro.py                   # GRPO + LoRA with <thinking> -> outputs/grpo-bbox-think (wandb "vlm-grpo-bbox")
python train_bbox_gpro.py --thinking false  # JSON-only variant      -> outputs/grpo-bbox
python eval_bbox.py --thinking --adapter outputs/grpo-bbox-think/checkpoint-100
python eval_bbox.py --adapter outputs/grpo-bbox/checkpoint-100
```

**Rewards** (both in `bbox_utils.py`):
- `detection_reward` (weight 1.0) is a soft F1. Predictions are matched to labels per class with Hungarian matching. Each matched pair with IoU ≥ 0.5 contributes its IoU. The reward is `2·ΣIoU / (n_pred + n_gt)`. Wrong labels, extra boxes and missed objects all lower it.
- `thinking_format_reward` (weight 0.25, thinking mode only) is 1.0 for a single non-empty `<thinking>…</thinking>` block before the answer. Thinking longer than 2,000 characters loses credit linearly and reaches 0 at 4,000.
- `format_reward` (weight 0.5) checks that the output is a clean JSON list with known labels and boxes inside [0, 1000]. The parser is lenient, so a truncated list still earns detection credit for the objects it finished.

**Setup.** LoRA r=64 is applied to the language-model linear layers; the vision tower is frozen. Each step uses 8 rollouts per image and 4 images, so 32 completions. Other settings:
- lr 2e-5, β=0 (no reference model), DAPO loss
- `max_completion_length` 3072: the longest ground-truth JSON is 2.0k tokens, and thinking needs room on top
- rollouts come from vLLM on the same GPU, which sleeps during backprop
- 5 epochs = 165 steps, about 3 min per step in thinking mode (about 8 h in total), peak VRAM 27.6 GB
- every 25 steps the eval split is scored with the rewards and a checkpoint is saved

**The BF16 base model is required.** vLLM can't take LoRA-merged weight syncs into FP8 layers, so training starts from a dequantized copy. `save_pretrained()` on a dequantized model re-quantizes to FP8, so the script copies the weights into a plain model first.

**vLLM workaround.** TRL 1.17 passes prompt token IDs that are already image-expanded. vLLM 0.32 expands Qwen3-VL images a second time (1,313 → 2,468 tokens). `patch_vllm_image_placeholders()` collapses each `<|image_pad|>` run before generation. TRL only uses vLLM's completion IDs, so training inputs are unchanged.

### Thinking mode

The goal is better detection: the model first reasons about which classes are present and where, then writes the boxes.

- **Prompt** (`PROMPT_THINK` in `bbox_utils.py`): scan the image region by region (top, left, center, right, bottom, rim) and note the classes, counts and locations in each region inside `<thinking></thinking>`. Then output only the JSON list after `</thinking>`.
- **Scoring:** only the JSON after `</thinking>` is scored. Boxes written inside the thinking earn nothing, and an unclosed or truncated `<thinking>` scores 0.
- **The tag is `<thinking>`, not Qwen's `<think>`,** so the chat template's reasoning handling doesn't touch it.
- **wandb:** the sample table has a separate `thinking` column. `think/well_formed` and `think/chars` track format compliance and thinking length.

Zero-shot, the base model already follows the format, so GRPO has a signal from step 0:

| eval split, greedy | thinking well-formed | P | R | F1 | soft F1 | strict JSON | truncated |
|---|---:|---:|---:|---:|---:|---:|---:|
| JSON only | – | 0.069 | 0.173 | 0.099 | 0.074 | 14% | 86% |
| `<thinking>` | 97% | 0.082 | 0.142 | 0.104 | 0.106 | 31% | 61% |

With sampling at temperature 1.0 on 32 train images, 91% of outputs are well-formed, the thinking averages about 540 characters, and nothing is truncated. The base model's thinking names classes per region but rarely gives counts or positions. That missing detail is what the detection reward should push it to add.

### Baseline accuracy (Qwen3-VL-4B-Instruct, zero-shot, greedy, eval split)

The scores are measured against SAM3 pseudo-labels at IoU ≥ 0.5. VLM outputs have no confidence scores, so P/R/F1 are reported instead of mAP.

| class | n_gt | n_pred | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| car | 320 | 540 | 0.207 | 0.350 | 0.260 |
| bus_truck | 45 | 35 | 0.543 | 0.422 | 0.475 |
| human | 306 | 1104 | 0.034 | 0.121 | 0.052 |
| bike | 481 | 1135 | 0.034 | 0.081 | 0.048 |
| traffic_sign | 12 | 157 | 0.000 | 0.000 | 0.000 |
| crosswalk | 49 | 54 | 0.056 | 0.061 | 0.058 |
| **all (micro)** | 1213 | 3025 | **0.069** | **0.173** | **0.099** |

Summary: macro F1 0.149, soft F1 (the reward) 0.074, strict JSON 14%, truncated at 3,072 tokens 86%.

The main failure is degenerate repetition under greedy decoding. After the first few objects the model counts boxes upward in a fixed step, for example `[880, 1460, …]`, `[880, 1480, …]` with y going past 1000. A third of the predicted boxes are duplicates. The boxes it does match are well placed (matched IoU about 0.70). Sampling at temperature 1.0 already avoids most loops: the train-set detection reward is about 0.17 at the first GRPO step.

## 4. Clock-reading data (TickTockVQA subset)

This is a small, simpler task for training reasoning VLMs: read the time on an analog clock. It comes from [TickTockVQA](https://huggingface.co/datasets/Jaeha-Choi/TickTockVQA) (CVPR 2026 Findings, arXiv 2603.08011).

```bash
python tools/prepare_clock_data.py   # -> data/clock/{train,eval}/*.jpg + data/clock/labels/{train,eval}.jsonl
```

- **Split:** 100 train images come from the dataset's *train* split and 20 eval images from its *test* split, so the two can't overlap. The seed is fixed. Use `--n_train` / `--n_eval` for more; there are 911 / 290 simple candidates (765 train after `--drop_1010`). A request larger than the pool is capped, and `-1` takes all of them.
- **Simple cases:** `transformation == Normal`, wall or alarm/desk clocks, Arabic numerals.
- **License:** only CC BY 4.0 rows (commercial use and redistribution allowed). Attribution fields are kept per image.
- **Label:** 12-hour `time` as `hh:mm`, plus `hour` (1–12) and `minute`. The dataset's AM/PM is kept as `ampm`, but a clock face can't show it, so it isn't part of the answer.

Caveats:
- Train is mostly CC12M product shots (large, centered clocks). Eval is COCO / OpenImages scenes, where the clock is sometimes small. Expect lower scores on eval.
- 16 of the 100 train clocks show about 10:10, the classic product-photo pose, so the hour distribution is skewed toward 10.
- The `design` field can list several numeral styles, so a few train clocks have Roman numerals. A few CC12M images carry stock-photo watermarks.

## 5. Clock-reading GRPO with thinking

```bash
python tools/prepare_clock_data.py          # 100 train / 20 eval (see section 4)
python eval_clock.py --thinking             # baseline
python train_clock_grpo.py                  # thinking mode -> outputs/grpo-clock-think, wandb project "vlm-grpo-clock"
python train_clock_grpo.py --thinking false # HH:MM only    -> outputs/grpo-clock
python eval_clock.py --thinking --adapter outputs/grpo-clock-think/checkpoint-100
```

- **Prompt** (`clock_utils.PROMPT_THINK`): inside `<thinking>`, find the hour hand (shorter) and the minute hand (longer), say what each points at, compute the minutes, check that the hour hand's position between two numbers agrees, then state the time. After `</thinking>`, output only `HH:MM`.
- **Rewards:**
  - `time_reward` (1.0): full credit within ±1 min on the 12-hour dial (wrapping at 12), falling linearly to 0 at 30 min off.
  - `format_reward` (0.25): the answer is exactly `HH:MM`.
  - `thinking_format_reward` (0.25).
- **Setup:** 4 images × 8 rollouts per step, 25 steps per epoch, 10 epochs = 250 steps. Steps take about 6–25 s (about 40 min in total), peak VRAM is 25 GB, and eval and saving happen every 25 steps.
- **Processor padding:** the processor must use left padding (`padding_side="left"`, as TRL's own default does). A right-padded processor misaligns Qwen3-VL's `mm_token_type_ids` whenever image sizes differ, and the run crashes with a `get_rope_index` shape mismatch.

Zero-shot baseline (eval split, greedy):

| | exact (±1 min) | within 5 min | hour correct | median error |
|---|---:|---:|---:|---:|
| HH:MM only | 25% | 40% | 45% | 38 min |
| `<thinking>` | 5% | 15% | 25% | 130 min |

Thinking *hurts* zero-shot. The model writes a "hour hand just past the 10, minute hand at the 2 → 10:10" story from its prior and answers 10:10 on 7 of 20 clocks. That is why `prepare_clock_data.py --drop_1010` (on by default) removes clocks near 10:10 from train: otherwise the prior gets rewarded.

## 6. Tracking how the thinking changes

`grpo_common.ThinkingProbeCallback` is on in both trainers. At step 0 and at every checkpoint save, the current policy answers a **fixed** set of eval images with greedy decoding: 8 for clock, 4 for bbox (`--n_probe`). Because the images and decoding are fixed, the outputs can be compared across steps. They are written to:
- `<output_dir>/checkpoint-N/thinking_probe.jsonl` (thinking, answer and score per image)
- `<output_dir>/thinking_probes.jsonl` (all steps, including step 0)
- wandb: the `thinking_probe` table and the `probe/score` and `probe/think_chars` scalars

Separately, the wandb `samples` table shows random training rollouts every 10 steps.

```bash
python tools/show_thinking.py outputs/grpo-clock-think                       # per-step summary + one image across steps
python tools/show_thinking.py outputs/grpo-clock-think --all --chars 0       # every probe image, full thinking
# older checkpoints without probes: run the eval on them first, it saves per-image thinking in the checkpoint dir
python eval_bbox.py --thinking --adapter outputs/grpo-bbox-think/checkpoint-50
python tools/show_thinking.py outputs/grpo-bbox-think --base outputs/eval/base_eval_think.jsonl
```

Backfilled bbox thinking run (`outputs/grpo-bbox-think`, eval split, greedy):

| step | P | R | F1 | soft F1 | truncated | boxes predicted (GT 1,213) | thinking chars |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (base) | 0.082 | 0.142 | 0.104 | 0.106 | 61% | 2092 | 665 |
| 25 | 0.100 | 0.188 | 0.131 | 0.128 | 56% | 2270 | 496 |
| 50 | 0.400 | 0.228 | 0.290 | 0.247 | 6% | 690 | 547 |
| 75 | 0.540 | 0.233 | 0.326 | 0.262 | 3% | 524 | 499 |

By step 50 the thinking turned from scene description into an explicit inventory ("3 cars, 1 bus, 1 human, no bikes"), and the repetition loops disappeared. Recall is flat at about 0.23. The policy became conservative: it predicts 524 boxes against 1,213 labels, and it often says "no bikes" where SAM3 labelled rows of parked scooters.

## 7. Clock GRPO results, 100 train images (`outputs/grpo-clock-think-100`)

The run used the `train_clock_grpo.py` defaults: 100 train images, 10 epochs = 250 steps, thinking mode. It was evaluated on the 20 eval images with greedy decoding:

```bash
python tools/eval_clock_checkpoints.py outputs/grpo-clock-think-100   # base + every checkpoint, one vLLM load
```

Full outputs are in `outputs/grpo-clock-think-100/thinking_progress/`:
- `report.md`: per image, the GT and the thinking and read time at every step
- `progress.csv` / `progress.jsonl`: thinking, read time and GT per step and image
- `metrics.csv`: metrics per step

`final` is identical to `checkpoint-250`.

| | base | step 50 | step 100 | step 150 | step 200 | final (250) |
|---|---|---|---|---|---|---|
| **exact (±1 min)** | 5% | 15% | 15% | **20%** | 15% | 0% |
| **within 5 min** | 20% | **35%** | 25% | **35%** | 25% | 0% |
| **hour correct** | 35% | 35% | **40%** | 35% | 30% | 15% |
| **median error (min)** | 134 | **56** | 68 | 59 | 58 | 108 |
| **mean reward** | 0.229 | **0.410** | 0.264 | 0.345 | 0.284 | 0.102 |
| **thinking chars (mean)** | 185 | 200 | 94 | 74 | 77 | 77 |
| **thinking, `clock_004` (GT 10:02)** | The clock face shows the hour hand pointing just past the 10 and the minute hand pointing at the 12, indicating 10:00. → **10:00** | Hour hand points just past the 10, minute hand points at the 12. Minute hand at 12 means 00 minutes. Hour hand slightly past 10 confirms 10:00. All consistent. → **10:00** | Hour hand between 10 and 11, closer to 10; minute hand on 12. Hour hand aligns with 12 minutes. → **10:55** | Hour hand between 11 and 12, closer to 12; minute hand on 12. Hour hand aligns. → **11:55** | Hour hand between 11 and 12, closer to 12; minute hand on 12. Hour hand aligns. → **11:53** | Hour hand between 11:59, closer to 11; minute hand on 11:59. Hour hand aligns. → **11:59** |
| **thinking, `clock_009` (GT 11:47)** | The clock face shows the hour hand pointing just past the 11 and the minute hand pointing at the 12, indicating 11:00. The small subdials for temperature and humidity are not relevant to the time. The red second hand is pointing near the 12, but the time is determined by the hour and minute hands. → **11:00** | Hour hand is just past the 12, minute hand is pointing at the 12, seconds hand is at 12. The hour hand is slightly past 12, consistent with 12:00. The small dials show no deviation from 12:00. All hands align with 12:00. → **12:00** | Hour hand between 12 and 1, closer to 12; minute hand on 12. Hour hand aligns with 12:00 minutes. → **11:47** ✅ | Hour hand between 12 and 1, closer to 12; minute hand on 12. Hour hand aligns. → **11:47** ✅ | Hour hand between 11 and 12, closer to 12; minute hand on 12. Hour hand aligns. → **11:46** ✅ | Hour hand between 11:59, closer to 11; minute hand on 11:59.59. Hour hand aligns. → **11:59:59** |

With 20 eval images, one image is 5 points, so steps 50–200 are roughly flat. The best checkpoint is around **step 150–175**: step 175 reaches exact 20%, within-5 30% and a median error of 44 min.

**How the thinking changed:**
1. **Base:** a free-form description driven by the prior. It reads 10:10 on 5 of 20 clocks, and the minute hand is almost always "at the 12".
2. **Step 50:** a longer, step-by-step check ("Minute hand at 12 means 00 minutes … confirms 10:00. All consistent."). It still reads mostly whole hours.
3. **Steps 100–200:** the thinking collapses into a fixed template, "Hour hand between X and Y, closer to X; minute hand on N. Hour hand aligns.", at about 75 characters. The 10:10 habit is gone.
4. **Final:** degenerate. The thinking parrots "11:59", and most answers are 11:58 or 11:59, sometimes the invalid `11:59:59`.

**The thinking does not support the answer.** From step 100 on, the minute hand is described as "on 12" even when the answer is 11:47 or 11:55. "Hour hand aligns" is a stock phrase, not a check that was actually done. The minutes in the answer come from somewhere other than the stated reasoning, so the thinking is not faithful.

**Why it collapsed after about step 175:** this is a training failure, not overfitting. The *training* reward fell from 1.10 (steps 151–175) to 0.79 (steps 226–250).
- **Entropy dropped:** from 0.25 to about 0.02 by step 100, and completions shrank from 64 to 37 tokens. The 8 rollouts per image became nearly identical, which leaves GRPO with almost no learning signal.
- **Nothing anchored the policy:** there is no KL term (β=0) and the learning rate stays constant at 2e-5, over 10 epochs of only 100 images.

**Next run:**
- more data: `tools/prepare_clock_data.py --n_train 500` (765 candidates are available)
- `--beta 0.02 --learning_rate 1e-5 --lr_scheduler_type cosine`, and fewer epochs
- a consistency reward that compares the hand positions stated in the thinking with the final answer

## 8. Clock GRPO results, 765 train images (`outputs/grpo-clock-think`)

All simple candidates were used for training, and the run went for 3 epochs:

```bash
python tools/prepare_clock_data.py --n_train -1          # 765 train (all simple candidates after --drop_1010) / 20 eval
python train_clock_grpo.py --num_train_epochs 3 --save_steps 50   # 4 images/step -> 191 steps/epoch, 573 steps, ~51 min
python tools/eval_clock_checkpoints.py outputs/grpo-clock-think --steps 50,150,300,450,573
```

Every other setting is the section 5 default: lr 2e-5 constant, β=0, 8 rollouts. Evaluation used the same 20 eval images with greedy decoding.

| | base | step 50 | step 150 | step 300 | step 450 | final (573) |
|---|---:|---:|---:|---:|---:|---:|
| **exact (±1 min)** | 5% | **20%** | 0% | 0% | 0% | 0% |
| **within 5 min** | 20% | **35%** | 0% | 0% | 0% | 0% |
| **hour correct** | 35% | **35%** | 20% | 15% | 5% | 20% |
| **median error (min)** | 134 | **58** | 110 | 170 | 176 | 74 |
| **thinking chars (mean)** | 185 | 34 | 0 | 0 | 0 | 0 |

Examples (GT → output):
- **`clock_004` (GT 10:02):**
  - base: `<thinking>…hour hand just past the 10 and the minute hand pointing at the 12, indicating 10:00.</thinking> 10:00`
  - step 50: `<thinking>11:00</thinking> 11:00`
  - final: `8:59`
- **`clock_009` (GT 11:47):** step 50 gives `<thinking>11:45</thinking> 11:45`, and the final model gives `11:59`.

**The thinking collapsed, and the result is worse than the 100-image run.** Step 50 is the best checkpoint and matches the best of section 7. After that, every checkpoint is below the base model.
1. **Steps 1–95:** the thinking shrinks from about 68 to about 12 tokens. By step 50 it only repeats the answer inside the tags.
2. **Steps 95–97:** the model drops `<thinking>` altogether and outputs a bare `H:MM` (about 5 tokens). The `thinking_format_reward` falls from 1.0 to 0 and never recovers. Its weight is only 0.25, so skipping the reasoning costs little.
3. **After the collapse:** 91% of groups get the same reward for all 8 rollouts (`frac_reward_zero_std`), compared with 57% before. The policy gets almost no gradient and stays stuck. Mean training reward falls from 0.88 (steps 1–95) to 0.41 (steps 97–573).

More data did not help, because the failure is in the reward design, not in the amount of data. `time_reward` is 0 beyond 30 min of error, and the base model's median error is 134 min. So most groups carry no signal, and the cheap way to raise reward is to shorten or drop the thinking.

**Next run:**
- **A denser time reward:** score the hour and minute hands separately, with the minute error taken mod 60, so a hand mix-up still earns partial credit.
- **Make thinking mandatory:** multiply the time reward by a "has thinking" gate, and give no credit for empty thinking (`<thinking>HH:MM</thinking>`).
- **Anchor the policy:** `--beta 0.02 --learning_rate 1e-5`, and save every 10–25 steps early on.
- **A bigger eval set:** `--n_eval 200` (290 candidates). With 20 images, one image moves every metric by 5 points.
- **More training data:** synthetic rendered clocks, used for a short supervised warm-up before GRPO.
