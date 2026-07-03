"""
Example: Energy-guided generation with Qwen3-0.6B-mdlm.

A lightweight demo that runs on any GPU with >2GB VRAM.
Good for development, testing, and quick iteration.

Usage:
    python examples/guided_story_qwen3.py
"""

import torch
import sys

from m2m_energy_fields import EnergyGuidedSampler, GuidanceConfig
from m2m_energy_fields.metrics import evaluate_guidance
from dllm.utils import get_model, get_tokenizer
from sentence_transformers import SentenceTransformer

DEVICE = "cuda"
MODEL_ID = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"


def clean_response(text: str) -> str:
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    if "<think>" in text and "</think>" in text:
        s = text.find("<think>")
        e = text.find("</think>") + len("</think>")
        text = text[:s] + text[e:]
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    return text


def main():
    print("=" * 60)
    print("m2m-energy-fields: Qwen3-0.6B Demo")
    print("=" * 60)

    model = get_model(
        model_args=type("Args", (), {
            "model_name_or_path": MODEL_ID,
            "dtype": torch.bfloat16,
            "device_map": {"": 0},
        })()
    ).eval()
    tokenizer = get_tokenizer(
        model_args=type("Args", (), {"model_name_or_path": MODEL_ID})()
    )
    evaluator = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=DEVICE
    )
    sampler = EnergyGuidedSampler(model=model, tokenizer=tokenizer)

    prompt = [{"role": "user", "content": "Write a short story. Make it interesting."}]
    config = GuidanceConfig(alpha=5.0, temperature=0.6)

    experiments = [
        ("baseline", None),
        ("horror",   ["horror nightmare monster ghost darkness fear terrifying"]),
        ("ocean",    ["ocean underwater coral reef fish diving deep sea"]),
    ]

    for name, target in experiments:
        print(f"\n{'─' * 60}")
        print(f"  {name.upper()}")
        print(f"{'─' * 60}")

        sampler.set_guidance(target_texts=target, alpha=config.alpha)
        if target:
            print(f"  Top tokens: {sampler.top_guided_tokens(8)}")

        inputs = tokenizer.apply_chat_template(
            [prompt], add_generation_prompt=True, tokenize=True
        )
        if isinstance(inputs[0], int):
            inputs = [inputs]

        with sampler:
            outputs = sampler.sample(inputs, config)

        for seq in outputs.sequences:
            response = clean_response(tokenizer.decode(seq, skip_special_tokens=False))
            metrics = evaluate_guidance(response, target, evaluator=evaluator)
            print(f"\n  {response[:250]}")
            if target:
                print(f"  target_sim={metrics['target_sim']}, coh={metrics['coherence']}")


if __name__ == "__main__":
    main()
