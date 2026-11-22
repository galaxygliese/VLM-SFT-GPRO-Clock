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
