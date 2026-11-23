"""GRPO training of Qwen3-VL-4B to read analog clocks (TickTockVQA subset in data/clock), with thinking + wandb.

The model reasons inside <thinking></thinking> (find the hands, what each points at, cross-check
hour-hand position vs minutes) and then answers HH:MM. Only the answer after </thinking> is scored.

Rewards (clock_utils.py):
  - time_reward (1.0):            1 within 1 min on the 12h dial, linear to 0 at 30 min off
  - format_reward (0.25):         answer is exactly HH:MM
  - thinking_format_reward (0.25): one non-empty leading <thinking> block (soft length cap)

Thinking progress: fixed probe images (first --n_probe eval images) are answered greedily at step 0
and at every save -> <ckpt>/thinking_probe.jsonl, <output_dir>/thinking_probes.jsonl, wandb
"thinking_probe" table. Compare checkpoints with: python tools/show_thinking.py outputs/grpo-clock-think

    python prepare: python tools/prepare_clock_data.py
    python train_clock_grpo.py
    python train_clock_grpo.py --thinking false --num_train_epochs 5
    python train_clock_grpo.py --synth_root data/clock_synth --n_synth 1000          # + synthetic clocks
    python train_clock_grpo.py --init_adapter outputs/sft-clock-think/final          # continue the SFT adapter
"""
import os
import sys
from dataclasses import dataclass, field

import torch
from datasets import Dataset, Features, Image, Value
from peft import LoraConfig
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, HfArgumentParser
from trl import GRPOConfig, GRPOTrainer

from utils.clock_utils import (format_reward, get_prompt, load_split, probe_score, split_thinking,
                         thinking_format_reward, time_reward, time_score)
from grpo_common import ThinkingProbeCallback, patch_vllm_image_placeholders


@dataclass
class ScriptArgs:
    model: str = field(default="checkpoints/Qwen3-VL-4B-Instruct-bf16",
                       metadata={"help": "BF16 base model (tools/dequantize_fp8.py)"})
    data_root: str = "data/clock"
    synth_root: str = field(default="", metadata={"help": "also train on clock_synth.py output, e.g. data/clock_synth"})
    n_synth: int = field(default=-1, metadata={"help": "synthetic train images to add (-1 = all)"})
    init_adapter: str = field(default="", metadata={"help": "start from this LoRA adapter (e.g. the SFT one) "
                                                            "instead of a fresh one; --lora_* are then ignored"})
    thinking: bool = field(default=True, metadata={"help": "reason in <thinking></thinking> before HH:MM"})
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    time_weight: float = 1.0
    format_weight: float = 0.25
    think_weight: float = 0.25
    n_probe: int = field(default=8, metadata={"help": "fixed eval images for the thinking probe (0=off)"})
    probe_max_new_tokens: int = 1024
    log_samples_every: int = field(default=10, metadata={"help": "wandb rollout samples every N steps (0=off)"})
    wandb_project: str = "vlm-grpo-clock"


DEFAULTS = [
    "--report_to", "wandb",
    "--bf16", "true",
    "--gradient_checkpointing", "true",
    # 4 images x 8 rollouts = 32 completions per optimizer step; 100 images -> 25 steps/epoch
    "--learning_rate", "2e-5",
    "--lr_scheduler_type", "constant_with_warmup",
    "--warmup_steps", "5",
    "--max_grad_norm", "1.0",
    "--num_train_epochs", "1",
    "--per_device_train_batch_size", "8",
    "--gradient_accumulation_steps", "4",
    "--num_generations", "8",
    "--beta", "0.0",
    "--loss_type", "dapo",
    "--max_completion_length", "1024",
    "--temperature", "1.0",
    "--use_vllm", "true",
    "--vllm_mode", "colocate",
    "--vllm_gpu_memory_utilization", "0.45",
    "--vllm_enable_sleep_mode", "true",
    "--vllm_max_model_length", "3072",
    "--logging_steps", "1",
    "--log_completions", "false",  # TRL's table fails on multimodal prompts in wandb; we log our own
    "--eval_strategy", "steps",
    "--eval_steps", "25",
    "--per_device_eval_batch_size", "8",
    "--num_generations_eval", "4",
    "--save_strategy", "steps",
    "--save_steps", "25",
    "--save_total_limit", "20",
    "--seed", "42",
]


def build_dataset(rows, prompt):
    return Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": prompt}] for _ in rows],
            "image": [r["image_path"] for r in rows],
            "hour": [r["hour"] for r in rows],
            "minute": [r["minute"] for r in rows],
            "time": [r["time"] for r in rows],
        },
        features=Features({
            "prompt": [{"role": Value("string"), "content": Value("string")}],
            "image": Image(),
            "hour": Value("int32"),
            "minute": Value("int32"),
            "time": Value("string"),
        }),
    )


def make_time_reward(every, n_show=8):
    """time_reward + every `every` steps a wandb table of rollouts (image, gt, thinking, answer, reward)."""
    last = {"step": -1}

    def time_reward_logged(completions, hour, minute, time=None, image=None, trainer_state=None, **kwargs):
        rewards = time_reward(completions, hour, minute, **kwargs)
        step = trainer_state.global_step if trainer_state is not None else 0
        if every and image is not None and step % every == 0 and step != last["step"]:
            last["step"] = step
            try:
                import wandb
                if wandb.run is not None:
                    table = wandb.Table(columns=["step", "image", "gt", "answer", "reward", "thinking"])
                    for i in range(min(n_show, len(completions))):
                        text = completions[i][0]["content"] if isinstance(completions[i], list) else completions[i]
                        thinking, answer, _ = split_thinking(text)
                        table.add_data(step, wandb.Image(image[i]), time[i] if time else "",
                                       answer.strip()[:50], float(rewards[i]), thinking[:4000])
                    wandb.log({"samples": table}, commit=False)
            except Exception as e:
                print(f"[sample logging failed] {e!r}")
        return rewards

    time_reward_logged.__name__ = "time_reward"
    return time_reward_logged


def main():
    parser = HfArgumentParser((ScriptArgs, GRPOConfig))
    script_args, cfg = parser.parse_args_into_dataclasses(args=DEFAULTS + sys.argv[1:])
    argv = " ".join(sys.argv[1:])
    if "--output_dir" not in argv:
        cfg.output_dir = "outputs/grpo-clock" + ("-think" if script_args.thinking else "")
    if "--run_name" not in argv:
        cfg.run_name = "qwen3vl4b-lora-grpo-clock" + ("-think" if script_args.thinking else "")
    cfg.model_init_kwargs = {"dtype": torch.bfloat16}
    os.environ.setdefault("WANDB_PROJECT", script_args.wandb_project)
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

    prompt = get_prompt(script_args.thinking)
    reward_funcs = [make_time_reward(script_args.log_samples_every), format_reward]
    cfg.reward_weights = [script_args.time_weight, script_args.format_weight]
    if script_args.thinking:
        reward_funcs.append(thinking_format_reward)
        cfg.reward_weights.append(script_args.think_weight)

    root = script_args.data_root
    train_rows = load_split(f"{root}/labels/train.jsonl", root)
    eval_rows = load_split(f"{root}/labels/eval.jsonl", root)
    n_real = len(train_rows)
    if script_args.synth_root:
        syn = load_split(f"{script_args.synth_root}/labels/train.jsonl", script_args.synth_root)
        train_rows += syn if script_args.n_synth < 0 else syn[:script_args.n_synth]
    print(f"train: {n_real} real + {len(train_rows) - n_real} synthetic = {len(train_rows)} | eval: {len(eval_rows)}")
    # left padding like TRL's own default: prompts are left-padded, and a right-padded processor would
    # misalign mm_token_type_ids (Qwen3-VL M-RoPE) whenever image sizes / prompt lengths differ
    processor = AutoProcessor.from_pretrained(script_args.model, padding_side="left")

    model, peft_config = script_args.model, LoraConfig(
        r=script_args.lora_r, lora_alpha=script_args.lora_alpha, lora_dropout=script_args.lora_dropout,
        task_type="CAUSAL_LM",
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    if script_args.init_adapter:  # keep training the given adapter (TRL accepts a PeftModel + peft_config=None)
        base = AutoModelForImageTextToText.from_pretrained(script_args.model, dtype=torch.bfloat16)
        model, peft_config = PeftModel.from_pretrained(base, script_args.init_adapter, is_trainable=True), None
        print(f"init adapter: {script_args.init_adapter}")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=cfg,
        train_dataset=build_dataset(train_rows, prompt),
        eval_dataset=build_dataset(eval_rows, prompt) if cfg.eval_strategy != "no" else None,
        processing_class=processor,
        peft_config=peft_config,
    )
    patch_vllm_image_placeholders(trainer)
    if script_args.n_probe:
        ThinkingProbeCallback(processor, eval_rows[:script_args.n_probe], prompt, probe_score,
                              max_new_tokens=script_args.probe_max_new_tokens, batch_size=8).attach(trainer)
    if trainer.accelerator.is_main_process and "wandb" in (cfg.report_to or []):
        import wandb
        if wandb.run is not None:
            wandb.config.update({"script": vars(script_args), "prompt": prompt}, allow_val_change=True)
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model(os.path.join(cfg.output_dir, "final"))


if __name__ == "__main__":
    main()
