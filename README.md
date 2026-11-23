# VLM-GPRO-BBox

## 1. Intro

Reinforcement learning (GRPO) and supervised fine-tuning (SFT) of a small vision-language model, **Qwen3-VL-4B-Instruct + LoRA**, on two visual tasks, with explicit reasoning in `<thinking>…</thinking>` before the answer:

- **Analog clock reading** (main task): read `HH:MM` from real photos ([TickTockVQA](https://huggingface.co/datasets/Jaeha-Choi/TickTockVQA)) and from synthetic clocks we render ourselves.
- **Bounding-box detection** on fisheye traffic-camera images ([FishEye8K](https://github.com/MoyoG/FishEye8K)) against SAM3 pseudo-labels.

Everything runs on one RTX 5090 (32 GB). Environment: `~/.dev_venv` (torch 2.13, transformers 5.19, TRL 1.17, peft 0.19, vLLM 0.32).

### Repository

| file | purpose |
|---|---|
| `tools/prepare_clock_data.py` | TickTockVQA subset → `data/clock/{train,eval}` (simple, upright, CC BY 4.0 clocks) |
| `clock_synth.py` | synthetic analog clocks with exact labels → `data/clock_synth` |
| `clock_utils.py` | clock prompt, time parsing, rewards, and `think_template()` (reference reasoning for SFT) |
| `train_clock_sft.py` | SFT warm-up (LoRA) on synthetic + real clocks |
| `train_clock_grpo.py` | GRPO (LoRA), optionally with synthetic data (`--synth_root`) and from the SFT adapter (`--init_adapter`) |
| `eval_clock.py`, `tools/eval_clock_checkpoints.py`, `tools/eval_clock_compare.py` | one model / every checkpoint of a run / several models side by side |
| `tools/make_labels.py`, `bbox_utils.py`, `train_bbox_gpro.py`, `eval_bbox.py` | bbox pseudo-labels, rewards, training and eval |
| `grpo_common.py` | vLLM image-placeholder patch, thinking-probe callback |
| `tools/show_thinking.py` | print how the thinking changes across checkpoints |

### Quick start (clock)

```bash
python tools/dequantize_fp8.py                                  # once: BF16 base -> checkpoints/Qwen3-VL-4B-Instruct-bf16
python tools/prepare_clock_data.py --n_train -1 --n_eval 100    # 765 real train / 100 real eval
python clock_synth.py --n 5000                                  # 5000 synthetic train clocks

python train_clock_sft.py                                       # -> outputs/sft-clock-think/final
python train_clock_grpo.py --synth_root data/clock_synth --n_synth 1000 \
    --init_adapter outputs/sft-clock-think/final --output_dir outputs/grpo-clock-think-sft

python tools/eval_clock_compare.py sft=outputs/sft-clock-think/final \
    sft_grpo=outputs/grpo-clock-think-sft/final                 # base + adapters, one vLLM load
```

### Setup notes

- **BF16 base model is required.** vLLM can't sync LoRA-merged weights into FP8 layers, so training starts from a dequantized copy (`tools/dequantize_fp8.py`).
- **vLLM image placeholders.** TRL 1.17 passes image-expanded prompt IDs, and vLLM 0.32 expands Qwen3-VL images a second time. `grpo_common.patch_vllm_image_placeholders()` collapses them before generation.
- **Left padding.** The processor must use `padding_side="left"`. Right padding misaligns Qwen3-VL's `mm_token_type_ids` when image sizes differ, and training crashes in `get_rope_index`.
- **Answer-only scoring.** Only the text after `</thinking>` is scored, and the tag is `<thinking>`, not Qwen's `<think>`, so the chat template leaves it alone.

## 2. Clock results

### Data

- **Real:** TickTockVQA "simple" clocks (upright, wall or alarm/desk, Arabic numerals, CC BY 4.0). Train has 765 images, mostly CC12M product shots, with clocks within 5 min of 10:10 removed (that pose is the base model's default guess). Eval has 100 images from COCO 2017 and Open Images scenes, where the clock is often small.
- **Synthetic:** 5,000 rendered clocks on random-colour backgrounds, with times uniform over the 12-hour dial. Faces, numerals (Arabic, Roman, 12/3/6/9 only, none), ticks, hand shapes, an optional second hand, rotation and blur all vary. Labels are exact.

### Training

| stage | data | steps | settings |
|---|---|---:|---|
| **SFT** | 5,000 synthetic + 765 real | 361 | LoRA r=64, lr 1e-4 cosine, 16 images/step, loss on completion only |
| **GRPO** (from the SFT adapter) | 1,000 synthetic + 765 real | 441 | 4 images × 8 rollouts/step, lr 2e-5 constant, β=0, DAPO loss, T=1.0 |

The SFT target is a reference reasoning generated from the label by `clock_utils.think_template()`. It follows the steps the prompt asks for. For 3:47:

> The hour hand (shorter) is about three quarters of the way from the 3 to the 4. The minute hand (longer) is 2 ticks past the 9, so 45 + 2 = 47 minutes. At 47 minutes the hour hand should be close to the 4 without reaching it, which matches, so the hour is still 3. The time is 3:47.

GRPO rewards: `time_reward` (1.0, full credit within ±1 min on the 12-hour dial, linear to 0 at 30 min off), `format_reward` (0.25, answer is exactly `HH:MM`) and `thinking_format_reward` (0.25, a non-empty `<thinking>` block).

### Results

All rows use the same 100 eval images with greedy decoding (`tools/eval_clock_compare.py`).

| model | exact (±1 min) | within 5 min | hour correct | median error |
|---|---:|---:|---:|---:|
| Base (zero-shot) | 9% | 17% | 28% | 73 min |
| **SFT** | 23% | **50%** | **59%** | **6 min** |
| SFT + GRPO, step 50 | 27% | 48% | 55% | 11 min |
| SFT + GRPO, step 250 | **30%** | **52%** | 55% | 4 min |
| SFT + GRPO, final (step 441) | 22% | 31% | 37% | 59 min |

Within-5-min accuracy by checkpoint:

```
base   SFT 100  SFT 200  SFT 361 | GRPO 50  GRPO 150  GRPO 250  GRPO 350  GRPO 441
 17%     42%      48%      50%   |   48%      37%       52%       32%       31%
```

How to read these numbers:
- With 100 images, one image is 1 point and 95% intervals are about ±10 points.
- **SFT is a clear gain.** In a paired bootstrap, base → SFT is +33 points within 5 min [+22, +43] and +15 exact [+7, +24].
- **GRPO after SFT does not reliably help.** Its checkpoints swing between 31% and 52%. Step 50 and step 250 match SFT within noise, and the final model is significantly worse. In a separate eval run where the final model scored 35% within 5 min, SFT → final was −15 points within 5 min [−26, −4] and −17 hour [−29, −5].
- Steps 50 and 250 were picked after seeing eval results, so their numbers are optimistic. Compare final checkpoints, or show the full curve.
- vLLM with LoRA isn't perfectly deterministic: repeated evals of the same model differ by 1–2 points.

### How the thinking changes

Measured on the 100 eval images:

| | Base | SFT | GRPO 50 | GRPO 250 | GRPO final |
|---|---:|---:|---:|---:|---:|
| reasoning counts minute ticks | 0 | 95 | 0 | 0 | 0 |
| answer is a multiple of 5 min | 97 | 5 | 100 | 100 | 53 |
| reasoning states "The time is …" | 0 | 100 | 100 | 0 | 0 |
| answer about 1 hour off | 23 | 11 | 16 | 18 | 27 |

1. **Base:** a short story from its prior. It answers 10:10 on 24 of 100 clocks.
2. **SFT:** the full template, including tick counting and the hour-hand check. The answer always matches the time stated in the reasoning.
3. **GRPO:** within 50 steps it stops counting ticks and rounds to 5 minutes. By step 250 the "The time is" conclusion is gone. At the end, the consistency check is self-contradicting boilerplate ("should be 3% of the way past the 11, but it's 18% of the way before it, so it matches"), and the answer sometimes disagrees with the reasoning.

A visual walkthrough of 4 eval clocks across all stages is in `outputs/thinking_evolution/thinking_timeline.html`.

### Why GRPO degrades: the reward

The two format rewards are always 1 after SFT, so `time_reward` is the only signal. It scores the answer and never the reasoning:

- **Small errors are almost free.** 2 minutes off still earns 0.97 and 4 minutes earns 0.90, so rounding to 5 minutes replaces tick counting.
- **An hour error scores 0,** the same as a random guess. 10:22 for 11:22 has the minutes exactly right but gets no credit, so nothing pushes the hour toward correct, and hour errors grow from 11 to 27.
- **The reasoning is unscored,** so it drifts until it no longer supports the answer.
- **Little signal per step, no anchor.** The SFT policy is nearly deterministic (entropy about 0.02), so in 75% of groups all 8 rollouts get the same reward. With β=0 and a constant learning rate, the few informative groups push the policy around, which explains the swinging curve.

An earlier GRPO run from the base model (765 real images, no SFT) collapsed differently. After about 95 steps it dropped the `<thinking>` block entirely and fell to 0% within 5 min. Losing the 0.25 thinking reward cost less than reasoning badly.

**Next:**
- Score the hour and minute separately (minute error measured on the 60-minute dial).
- Add a consistency reward: the time stated in the reasoning must equal the answer.
- `--beta 0.02–0.04` and a decaying learning rate.
- More diverse rollouts (T=1.2 or 16 per image).

## 3. Bbox results

Six classes (`car`, `bus_truck`, `human`, `bike`, `traffic_sign`, `crosswalk`) on FishEye8K images: 133 train and 36 eval. Labels are SAM3 pseudo-labels from text prompts (`tools/make_labels.py`). The reward is a soft F1: Hungarian matching per class at IoU ≥ 0.5, summing IoU.

GRPO with thinking (`outputs/grpo-bbox-think`), eval split, greedy:

| step | P | R | F1 | truncated | boxes predicted (GT 1,213) |
|---:|---:|---:|---:|---:|---:|
| 0 (base) | 0.082 | 0.142 | 0.104 | 61% | 2,092 |
| 50 | 0.400 | 0.228 | 0.290 | 6% | 690 |
| 75 | **0.540** | **0.233** | **0.326** | 3% | 524 |

F1 triples, mostly from precision. The base model's repetition loops (boxes counting upward until y > 1000) disappear, and the thinking becomes an explicit inventory ("3 cars, 1 bus, no bikes"). Recall stays at about 0.23: the policy predicts fewer than half as many boxes as there are labels.

## 4. Citations

**Model and methods**
- Qwen Team. *Qwen3-VL Technical Report.* arXiv:2511.21631, 2025.
- Z. Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO). arXiv:2402.03300, 2024.
- Q. Yu et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* arXiv:2503.14476, 2025.
- E. J. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. arXiv:2106.09685.
- N. Carion et al. *SAM 3: Segment Anything with Concepts.* ICLR 2026. arXiv:2511.16719.
- W. Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM). SOSP 2023. arXiv:2309.06180.
- L. von Werra et al. *TRL: Transformer Reinforcement Learning.* https://github.com/huggingface/trl

**Datasets**
- J. Choi, J. W. Lee, S. You, J. Lee. *It's Time to Get It Right: Improving Analog Clock Reading and Clock-Hand Spatial Reasoning in Vision-Language Models* (TickTockVQA). CVPR 2026 Findings. arXiv:2603.08011.
- M. Gochoo et al. *FishEye8K: A Benchmark and Dataset for Fisheye Camera Object Detection.* CVPR Workshops 2023. arXiv:2305.17449.
- Sources of the TickTockVQA images used here, all CC BY 4.0:
  - T.-Y. Lin et al. *Microsoft COCO: Common Objects in Context.* ECCV 2014. arXiv:1405.0312.
  - A. Kuznetsova et al. *The Open Images Dataset V4.* IJCV 2020. arXiv:1811.00982.
  - S. Changpinyo et al. *Conceptual 12M.* CVPR 2021. arXiv:2102.08981.
  - R. Krishna et al. *Visual Genome.* IJCV 2017. arXiv:1602.07332.

Clock images keep their per-image attribution fields (`source`, `source_url`, `license`) in `data/clock/labels/*.jsonl`. Credit them when you redistribute or publish them.
