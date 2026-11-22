"""Save a BF16 copy of an FP8 Qwen3-VL checkpoint (for training + vLLM weight sync).

    python tools/dequantize_fp8.py --src Qwen/Qwen3-VL-4B-Instruct-FP8 --dst checkpoints/Qwen3-VL-4B-Instruct-bf16
"""
import argparse

import torch
from transformers import AutoProcessor, FineGrainedFP8Config, Qwen3VLForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="Qwen/Qwen3-VL-4B-Instruct-FP8")
ap.add_argument("--dst", default="checkpoints/Qwen3-VL-4B-Instruct-bf16")
args = ap.parse_args()

model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.src, dtype=torch.bfloat16, device_map="cpu",
    quantization_config=FineGrainedFP8Config(dequantize=True))
sd = model.state_dict()
assert all(v.dtype != torch.float8_e4m3fn for v in sd.values()), "dequantization failed"
cfg = model.config
for c in (cfg, getattr(cfg, "text_config", None), getattr(cfg, "vision_config", None)):
    if c is not None and hasattr(c, "quantization_config"):
        delattr(c, "quantization_config")
# save_pretrained() on the dequantized model would re-quantize to FP8 -> copy into a plain model first
clean = Qwen3VLForConditionalGeneration._from_config(cfg, dtype=torch.bfloat16)
missing, unexpected = clean.load_state_dict(sd, strict=False)
assert not unexpected and all("lm_head" in k for k in missing), (missing, unexpected)
clean.tie_weights()
clean.save_pretrained(args.dst, safe_serialization=True)
AutoProcessor.from_pretrained(args.src).save_pretrained(args.dst)
print("saved", args.dst)
