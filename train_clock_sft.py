"""SFT warm-up for clock reading (Qwen3-VL-4B + LoRA), before GRPO (train_clock_grpo.py).

GRPO can only reinforce what the policy already does sometimes; zero-shot the model reads ~5% of clocks
exactly and its <thinking> is a 10:10 story from its prior. This stage teaches the reasoning format with
*faithful* reference thinking: clock_utils.think_template(hour, minute) describes where each hand must
point for the labelled time (hour hand position, minute hand + ticks, a consistency check), then HH:MM.

Data (prompt = clock_utils.PROMPT_THINK, the same prompt GRPO and the evals use):
  - synthetic clocks from clock_synth.py (uniform times, exact labels): data/clock_synth/labels/train.jsonl
  - real TickTockVQA train clocks (data/clock/labels/train.jsonl), repeated --real_repeat times
  - eval loss on data/clock/labels/eval.jsonl
Loss is on the completion (thinking + answer) only.

    python clock_synth.py --n 5000
    python train_clock_sft.py                                   # -> outputs/sft-clock-think/final (LoRA adapter)
    python eval_clock.py --thinking --adapter outputs/sft-clock-think/final
"""
import os
import random
import sys
from dataclasses import dataclass, field

import torch
from datasets import Dataset, Features, Image, Value
from peft import LoraConfig
from transformers import AutoProcessor, HfArgumentParser
from trl import SFTConfig, SFTTrainer

from clock_utils import get_prompt, load_split, think_template
import wandb

@dataclass
class ScriptArgs:
    model: str = field(default="checkpoints/Qwen3-VL-4B-Instruct-bf16",
                       metadata={"help": "BF16 base model (tools/dequantize_fp8.py)"})
    synth_root: str = field(default="data/clock_synth", metadata={"help": "clock_synth.py output ('' = none)"})
    n_synth: int = field(default=-1, metadata={"help": "synthetic examples to use (-1 = all)"})
    real_root: str = field(default="data/clock", metadata={"help": "prepare_clock_data.py output ('' = none)"})
    real_repeat: int = field(default=1, metadata={"help": "times each real train clock is repeated"})
    thinking: bool = field(default=True, metadata={"help": "target '<thinking>...</thinking>\\nHH:MM' (else HH:MM)"})
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    wandb_project: str = "vlm-sft-clock"
    seed_data: int = 0


DEFAULTS = [
    "--report_to", "wandb",
    "--bf16", "true",
    "--gradient_checkpointing", "true",
    "--learning_rate", "1e-4",
    "--lr_scheduler_type", "cosine",
    "--warmup_steps", "0.03",       # <1 = fraction of total steps (transformers 5)
    "--num_train_epochs", "1",
    "--per_device_train_batch_size", "8",
    "--gradient_accumulation_steps", "2",
    "--per_device_eval_batch_size", "8",
    "--completion_only_loss", "true",
    "--logging_steps", "10",
    "--eval_strategy", "steps",
    "--eval_steps", "100",
    "--save_strategy", "steps",
    "--save_steps", "100",
    "--save_total_limit", "5",
    "--dataloader_num_workers", "4",
    "--seed", "42",
]


def target(row, thinking, rng):
    if not thinking:
        return row["time"]
    return f"<thinking>{think_template(row['hour'], row['minute'], rng)}</thinking>\n{row['time']}"


def build_dataset(rows, prompt, thinking, rng):
    return Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": [{"type": "image", "text": None},
                                                     {"type": "text", "text": prompt}]}] for _ in rows],
            "completion": [[{"role": "assistant", "content": [{"type": "text", "text": target(r, thinking, rng)}]}]
                           for r in rows],
            "image": [r["image_path"] for r in rows],
        },
        features=Features({
            "prompt": [{"role": Value("string"), "content": [{"type": Value("string"), "text": Value("string")}]}],
            "completion": [{"role": Value("string"),
                            "content": [{"type": Value("string"), "text": Value("string")}]}],
            "image": Image(),
        }),
    )


def main():
    parser = HfArgumentParser((ScriptArgs, SFTConfig))
    script_args, cfg = parser.parse_args_into_dataclasses(args=DEFAULTS + sys.argv[1:])
    argv = " ".join(sys.argv[1:])
    if "--output_dir" not in argv:
        cfg.output_dir = "outputs/sft-clock" + ("-think" if script_args.thinking else "")
    if "--run_name" not in argv:
        cfg.run_name = "qwen3vl4b-lora-sft-clock" + ("-think" if script_args.thinking else "")
    if "--max_length" not in argv:
        cfg.max_length = None  # never truncate image tokens
    cfg.model_init_kwargs = {"dtype": torch.bfloat16}
    os.environ.setdefault("WANDB_PROJECT", script_args.wandb_project)

    rng = random.Random(script_args.seed_data)
    train_rows, n_syn, n_real = [], 0, 0
    if script_args.synth_root:
        syn = load_split(f"{script_args.synth_root}/labels/train.jsonl", script_args.synth_root)
        if script_args.n_synth >= 0:
            syn = syn[:script_args.n_synth]
        train_rows += syn
        n_syn = len(syn)
    if script_args.real_root and script_args.real_repeat > 0:
        real = load_split(f"{script_args.real_root}/labels/train.jsonl", script_args.real_root)
        train_rows += real * script_args.real_repeat
        n_real = len(real)
    rng.shuffle(train_rows)
    eval_rows = load_split(f"{script_args.real_root or 'data/clock'}/labels/eval.jsonl",
                           script_args.real_root or "data/clock")
    print(f"train: {n_syn} synthetic + {n_real} real x{script_args.real_repeat} = {len(train_rows)} | "
          f"eval: {len(eval_rows)} real")

    prompt = get_prompt(script_args.thinking)
    train_ds = build_dataset(train_rows, prompt, script_args.thinking, rng)
    eval_ds = build_dataset(eval_rows, prompt, script_args.thinking, rng) if cfg.eval_strategy != "no" else None
    print("example target:\n" + train_ds[0]["completion"][0]["content"][0]["text"])

    # left padding, as in train_clock_grpo.py: right padding misaligns Qwen3-VL's mm_token_type_ids
    # (M-RoPE) whenever image sizes differ within a batch
    processor = AutoProcessor.from_pretrained(script_args.model, padding_side="left")
    trainer = SFTTrainer(
        model=script_args.model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        peft_config=LoraConfig(
            r=script_args.lora_r, lora_alpha=script_args.lora_alpha, lora_dropout=script_args.lora_dropout,
            task_type="CAUSAL_LM",
            target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"),
    )
    if trainer.accelerator.is_main_process and "wandb" in (cfg.report_to or []):
        if wandb.run is not None:
            wandb.config.update({"script": vars(script_args), "prompt": prompt}, allow_val_change=True)
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    final = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final)
    processor.save_pretrained(final)
    print(f"adapter -> {final}")


if __name__ == "__main__":
    main()
