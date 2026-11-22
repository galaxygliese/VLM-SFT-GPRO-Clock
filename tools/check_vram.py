"""Check whether a Qwen3-VL model fits GRPO + LoRA training on this GPU.

Loads the model in BF16 (FP8 checkpoints are dequantized at load time), runs one zero-shot
grounding generation, then simulates a GRPO policy step: G completions of L tokens for one
image, LoRA on all linear layers, gradient checkpointing. Prints peak VRAM.

    python tools/check_vram.py --model Qwen/Qwen3-VL-4B-Instruct-FP8 --G 8 --L 512
"""
import argparse
import time

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, FineGrainedFP8Config, Qwen3VLForConditionalGeneration

PROMPT = ("Detect every car, bus/truck, person, bike, traffic light, traffic sign in the image. "
          'Output a JSON list of {"bbox_2d": [x1, y1, x2, y2], "label": ...}.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct-FP8")
    ap.add_argument("--image", default="data/fisheye8k/train/18_60sec_frame_00814.png")
    ap.add_argument("--G", type=int, default=8, help="completions per prompt (GRPO group size)")
    ap.add_argument("--L", type=int, default=512, help="completion length in tokens")
    ap.add_argument("--max_pixels", type=int, default=1024 * 1024)
    ap.add_argument("--lora_r", type=int, default=64)
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(args.model)
    proc.image_processor.max_pixels = args.max_pixels
    kw = {"quantization_config": FineGrainedFP8Config(dequantize=True)} if "FP8" in args.model else {}
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", **kw)
    print(f"weights: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=[text], images=[Image.open(args.image)], return_tensors="pt").to("cuda")

    model.eval()
    t = time.time()
    out = model.generate(**inp, max_new_tokens=300, do_sample=False)
    print(f"zero-shot generate ({time.time() - t:.1f}s):")
    print(proc.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True)[:800])

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.train()
    torch.cuda.reset_peak_memory_stats()

    G, L = args.G, args.L
    ids = torch.cat([inp.input_ids.repeat(G, 1),
                     torch.randint(1000, 50000, (G, L), device="cuda")], 1)
    mm = torch.cat([inp.mm_token_type_ids.repeat(G, 1),
                    torch.zeros(G, L, dtype=inp.mm_token_type_ids.dtype, device="cuda")], 1)
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=mm,
                 pixel_values=inp.pixel_values.repeat(G, 1), image_grid_thw=inp.image_grid_thw.repeat(G, 1))
    t = time.time()
    loss = model(**batch).logits[:, -L:].float().logsumexp(-1).mean()
    loss.backward()
    print(f"prompt={inp.input_ids.shape[1]} tok, G={G}, L={L}: fwd+bwd {time.time() - t:.1f}s, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
          f"/ {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
