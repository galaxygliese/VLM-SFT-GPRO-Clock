"""GRPO training of Qwen3-VL-4B for fisheye object detection (bbox JSON output), with wandb.

Thinking mode (default, --thinking true): the model first reasons inside <thinking></thinking> about
which classes are where (region by region), then outputs the JSON. Only the JSON after </thinking> is
scored. --thinking false trains the plain JSON-only task.

Rewards (bbox_utils.py):
  - detection_reward:       soft F1 @ IoU 0.5 vs the SAM3 pseudo-labels (per-class Hungarian matching)
  - format_reward:          clean JSON list with valid labels and boxes in [0, 1000]
  - thinking_format_reward: (thinking mode) one non-empty leading <thinking> block, soft length cap

Policy: LoRA on the language model. Rollouts: vLLM colocated on the same GPU, sleeping during the
backward pass. Every GRPOConfig field can be overridden on the command line:

    python train_bbox_gpro.py                       # thinking mode -> outputs/grpo-bbox-think
    python train_bbox_gpro.py --thinking false      # JSON only     -> outputs/grpo-bbox
    python train_bbox_gpro.py --num_train_epochs 10 --learning_rate 1e-5 --run_name my-run
    python train_bbox_gpro.py --max_steps 2 --report_to none --output_dir outputs/smoke   # smoke test
"""
import json
import os
import sys
from dataclasses import dataclass, field

import torch
from datasets import Dataset, Features, Image, Value
from peft import LoraConfig
from transformers import HfArgumentParser
from trl import GRPOConfig, GRPOTrainer

from bbox_utils import (CLASSES, answer_text, completion_text, detection_reward, format_reward, get_prompt,
                        load_split, parse_objects, split_thinking, thinking_format_reward)


@dataclass
class ScriptArgs:
    model: str = field(default="checkpoints/Qwen3-VL-4B-Instruct-bf16",
                       metadata={"help": "BF16 base model (tools/dequantize_fp8.py)"})
    data_root: str = "data/fisheye8k"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    thinking: bool = field(default=True, metadata={"help": "reason in <thinking></thinking> before the JSON"})
    format_weight: float = 0.5
    think_weight: float = 0.25
    detection_weight: float = 1.0
    wandb_project: str = "vlm-grpo-bbox"
    log_samples_every: int = field(default=10, metadata={"help": "wandb box-overlay samples every N steps (0=off)"})


# Defaults; anything passed on the command line comes later and wins.
DEFAULTS = [
    "--report_to", "wandb",
    "--bf16", "true",
    "--gradient_checkpointing", "true",
    # optimisation: 4 images x 8 rollouts = 32 completions per optimizer step
    "--learning_rate", "2e-5",
    "--lr_scheduler_type", "constant_with_warmup",
    "--warmup_steps", "5",
    "--max_grad_norm", "1.0",
    "--num_train_epochs", "5",
    "--per_device_train_batch_size", "2",
    "--gradient_accumulation_steps", "16",
    "--num_generations", "8",
    "--beta", "0.0",
    "--loss_type", "dapo",
    # generation
    "--max_completion_length", "3072",  # GT JSON is up to ~2.0k tokens, plus thinking
    "--temperature", "1.0",
    "--use_vllm", "true",
    "--vllm_mode", "colocate",
    "--vllm_gpu_memory_utilization", "0.5",
    "--vllm_enable_sleep_mode", "true",
    "--vllm_max_model_length", "4608",  # ~1.3k prompt (1156 image tokens) + completion
    # logging / eval / checkpoints
    "--logging_steps", "1",
    # TRL's completion table fails on multimodal prompts in wandb; we log our own (see make_detection_reward)
    "--log_completions", "false",
    "--eval_strategy", "steps",
    "--eval_steps", "25",
    "--per_device_eval_batch_size", "8",
    "--num_generations_eval", "2",
    "--save_strategy", "steps",
    "--save_steps", "25",
    "--save_total_limit", "10",
    "--seed", "42",
]


def build_dataset(path, prompt):
    rows = load_split(path, CLASSES)
    root = os.path.dirname(os.path.dirname(path))
    ds = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": prompt}] for _ in rows],
            "image": [os.path.join(root, r["image"]) for r in rows],
            "objects": [json.dumps(r["objects"]) for r in rows],
        },
        features=Features({
            "prompt": [{"role": Value("string"), "content": Value("string")}],
            "image": Image(),
            "objects": Value("string"),
        }),
    )
    return ds


def make_detection_reward(every, n_show=4):
    """detection_reward + every `every` steps a wandb table: image with pred (red) / GT (green) boxes."""
    last = {"step": -1}

    def detection_reward_logged(completions, objects, image=None, trainer_state=None, **kwargs):
        rewards = detection_reward(completions, objects, **kwargs)
        step = trainer_state.global_step if trainer_state is not None else 0
        if every and image is not None and step % every == 0 and step != last["step"]:
            last["step"] = step
            try:
                import wandb
                if wandb.run is not None:
                    _log_samples(wandb, step, completions, objects, image, rewards, n_show)
            except Exception as e:  # logging must never kill training
                print(f"[sample logging failed] {e!r}")
        return rewards

    detection_reward_logged.__name__ = "detection_reward"
    return detection_reward_logged


def _log_samples(wandb, step, completions, objects, images, rewards, n_show):
    cls_ids = {c: i for i, c in enumerate(CLASSES)}
    table = wandb.Table(columns=["step", "image", "reward", "n_pred", "n_gt", "thinking", "completion"])
    for i in range(min(n_show, len(completions))):
        img = images[i]
        gt = json.loads(objects[i])
        pred, _ = parse_objects(answer_text(completions[i]))
        thinking = split_thinking(completion_text(completions[i]))[0]

        def boxes(objs, prefix):
            out = []
            for o in objs:
                x1, y1, x2, y2 = (min(max(v / 1000, 0.0), 1.0) for v in o["bbox"])
                if x2 <= x1 or y2 <= y1:
                    continue
                out.append({"position": {"minX": x1, "minY": y1, "maxX": x2, "maxY": y2},
                            "class_id": cls_ids.get(o["label"], len(CLASSES)),
                            "box_caption": f"{prefix}{o['label']}"})
            return out

        labels = {**{i_: c for c, i_ in cls_ids.items()}, len(CLASSES): "unknown"}
        wimg = wandb.Image(img, boxes={
            "pred": {"box_data": boxes(pred, ""), "class_labels": labels},
            "gt": {"box_data": boxes(gt, "gt:"), "class_labels": labels}})
        table.add_data(step, wimg, float(rewards[i]), len(pred), len(gt), thinking[:4000],
                       completion_text(completions[i])[:6000])
    wandb.log({"samples": table}, commit=False)


def patch_vllm_image_placeholders(trainer):
    """Work around double image expansion (TRL 1.17 + vLLM 0.32 + Qwen3-VL).

    TRL sends prompt_token_ids that the HF processor already expanded (1 image -> 1156
    <|image_pad|> tokens). vLLM does not detect the expansion and inserts the image tokens a
    second time, which roughly doubles the prompt and misaligns the vision features. Collapsing
    each run of <|image_pad|> to a single token makes vLLM rebuild exactly the HF token sequence.
    TRL only uses the completion ids that vLLM returns, so training inputs are unaffected.
    """
    vg = getattr(trainer, "vllm_generation", None)
    if vg is None or getattr(vg, "llm", None) is None:
        return
    tok = trainer.processing_class.tokenizer
    pad_ids = {tok.convert_tokens_to_ids(t) for t in ("<|image_pad|>", "<|video_pad|>")}
    llm, orig_generate = vg.llm, vg.llm.generate

    def generate(prompts, *args, **kwargs):
        fixed = []
        for p in prompts:
            if isinstance(p, dict) and "prompt_token_ids" in p and "multi_modal_data" in p:
                ids = p["prompt_token_ids"]
                ids = [t for i, t in enumerate(ids) if not (t in pad_ids and i > 0 and ids[i - 1] == t)]
                p = {**p, "prompt_token_ids": ids}
            fixed.append(p)
        return orig_generate(fixed, *args, **kwargs)

    llm.generate = generate


def main():
    parser = HfArgumentParser((ScriptArgs, GRPOConfig))
    script_args, cfg = parser.parse_args_into_dataclasses(args=DEFAULTS + sys.argv[1:])
    argv = " ".join(sys.argv[1:])
    if "--output_dir" not in argv:
        cfg.output_dir = "outputs/grpo-bbox-think" if script_args.thinking else "outputs/grpo-bbox"
    if "--run_name" not in argv:
        cfg.run_name = "qwen3vl4b-lora-grpo-bbox" + ("-think" if script_args.thinking else "")
    prompt = get_prompt(script_args.thinking)
    reward_funcs = [make_detection_reward(script_args.log_samples_every), format_reward]
    cfg.reward_weights = [script_args.detection_weight, script_args.format_weight]
    if script_args.thinking:
        reward_funcs.append(thinking_format_reward)
        cfg.reward_weights.append(script_args.think_weight)
    cfg.model_init_kwargs = {"dtype": torch.bfloat16}
    os.environ.setdefault("WANDB_PROJECT", script_args.wandb_project)
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

    train_ds = build_dataset(f"{script_args.data_root}/labels/train.jsonl", prompt)
    eval_ds = build_dataset(f"{script_args.data_root}/labels/eval.jsonl", prompt) if cfg.eval_strategy != "no" else None

    peft_config = LoraConfig(
        r=script_args.lora_r, lora_alpha=script_args.lora_alpha, lora_dropout=script_args.lora_dropout,
        task_type="CAUSAL_LM",
        # language-model linears only (the vision tower uses qkv/proj/linear_fc* names)
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    )

    trainer = GRPOTrainer(
        model=script_args.model,
        reward_funcs=reward_funcs,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )
    patch_vllm_image_placeholders(trainer)
    if trainer.accelerator.is_main_process and "wandb" in (cfg.report_to or []):
        import wandb
        if wandb.run is not None:
            wandb.config.update({"script": vars(script_args), "classes": CLASSES, "prompt": prompt},
                                allow_val_change=True)
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model(os.path.join(cfg.output_dir, "final"))


if __name__ == "__main__":
    main()
