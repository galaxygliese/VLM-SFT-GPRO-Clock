"""Pieces shared by the GRPO trainers (bbox and clock).

- patch_vllm_image_placeholders: fix for double image expansion (TRL 1.17 + vLLM 0.32 + Qwen3-VL)
- ThinkingProbeCallback: at step 0 and at every checkpoint save, answer a FIXED set of eval images
  greedily and record thinking / answer / score, so the evolution of the thinking is comparable
  across checkpoints. Written to <ckpt>/thinking_probe.jsonl, <output_dir>/thinking_probes.jsonl
  and wandb (table "thinking_probe", scalar "probe/score").
"""
import json
import os
import time

import torch
from PIL import Image
from transformers import TrainerCallback


def patch_vllm_image_placeholders(trainer):
    """Work around double image expansion (TRL 1.17 + vLLM 0.32 + Qwen3-VL).

    TRL sends prompt_token_ids that the HF processor already expanded (1 image -> N
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


class ThinkingProbeCallback(TrainerCallback):
    """Greedy answers on fixed probe images at step 0 and at every save.

    probes:   list of dicts with at least {"image_path": str, "id": str}; passed to score_fn
    prompt:   user prompt text
    score_fn: (completion_text, probe) -> dict with at least "score" (float), plus any extra fields
              (e.g. "thinking", "answer") to record
    """

    def __init__(self, processor, probes, prompt, score_fn, max_new_tokens=1024, batch_size=4):
        self.processor = processor
        self.probes = probes
        self.prompt = prompt
        self.score_fn = score_fn
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.trainer = None  # set by attach()

    def attach(self, trainer):
        self.trainer = trainer
        trainer.add_callback(self)
        return self

    @torch.no_grad()
    def _generate(self, model):
        proc = self.processor
        tok = proc.tokenizer
        old_side = tok.padding_side
        tok.padding_side = "left"
        was_training = model.training
        model.eval()
        texts = []
        try:
            for i in range(0, len(self.probes), self.batch_size):
                chunk = self.probes[i:i + self.batch_size]
                msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]}]
                        for _ in chunk]
                chat = [proc.apply_chat_template(m, add_generation_prompt=True) for m in msgs]
                imgs = [Image.open(p["image_path"]).convert("RGB") for p in chunk]
                inp = proc(text=chat, images=imgs, return_tensors="pt", padding=True).to(model.device)
                out = model.generate(**inp, max_new_tokens=self.max_new_tokens, do_sample=False, use_cache=True)
                for row in out[:, inp["input_ids"].shape[1]:]:
                    texts.append(tok.decode(row, skip_special_tokens=True))
        finally:
            tok.padding_side = old_side
            if was_training:
                model.train()
        return texts

    def _run(self, args, state, model, ckpt_dir=None):
        if not state.is_world_process_zero or model is None:
            return
        t = time.time()
        texts = self._generate(model)
        rows = []
        for p, text in zip(self.probes, texts):
            rec = self.score_fn(text, p)
            rows.append({"step": state.global_step, "id": p["id"], **rec, "completion": text})
        mean = sum(r["score"] for r in rows) / max(len(rows), 1)
        print(f"[thinking probe] step {state.global_step}: mean score {mean:.3f} ({time.time() - t:.0f}s)")

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "thinking_probes.jsonl"), "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        if ckpt_dir and os.path.isdir(ckpt_dir):
            with open(os.path.join(ckpt_dir, "thinking_probe.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        try:
            import wandb
            if wandb.run is not None:
                cols = ["step", "id", "score", "thinking", "answer", "completion"]
                table = wandb.Table(columns=cols)
                for r in rows:
                    table.add_data(r["step"], r["id"], r["score"], r.get("thinking", ""), r.get("answer", ""),
                                   r["completion"][:6000])
                wandb.log({"thinking_probe": table, "probe/score": mean,
                           "probe/think_chars": sum(len(r.get("thinking", "")) for r in rows) / max(len(rows), 1)},
                          commit=False)
        except Exception as e:  # logging must never kill training
            print(f"[thinking probe] wandb logging failed: {e!r}")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0:
            self._run(args, state, model)

    def on_save(self, args, state, control, model=None, **kwargs):
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        self._run(args, state, model, ckpt_dir=ckpt)
