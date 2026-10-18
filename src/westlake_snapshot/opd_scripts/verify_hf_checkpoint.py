#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    weights = checkpoint / "model.safetensors"
    with safe_open(weights, framework="pt", device="cpu") as handle:
        tensor_keys = list(handle.keys())
        stored_numel = sum(math.prod(handle.get_slice(key).get_shape()) for key in tensor_keys)

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        device_map={"": "meta"},
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model_numel = sum(parameter.numel() for parameter in model.parameters())

    print(
        "HF_LOAD_OK",
        f"class={model.__class__.__name__}",
        f"model_numel={model_numel}",
        f"stored_numel={stored_numel}",
        f"tensor_keys={len(tensor_keys)}",
        f"vocab_size={model.config.vocab_size}",
        f"tokenizer_size={len(tokenizer)}",
        f"device={next(model.parameters()).device}",
    )


if __name__ == "__main__":
    main()
