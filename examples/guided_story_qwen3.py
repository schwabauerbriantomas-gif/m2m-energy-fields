"""
Example: Energy-guided generation with Qwen3-0.6B-mdlm.

A lightweight demo that runs on any GPU with >2GB VRAM.
Good for development, testing, and quick iteration.

Usage:
    python examples/guided_story_qwen3.py
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from m2m_energy_fields import EnergyGuidedSampler, GuidanceConfig
from m2m_energy_fields.metrics import evaluate_guidance
from dllm.utils import get_model, get_tokenizer

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
    sampler = EnergyGuidedSampler(model=model, tokenizer=tokenizer)

    prompt = [{"role": "user", "content": "Write a short story. Make it interesting."}]
    config = GuidanceConfig(
        alpha=10.0, temperature=0.6, steps=64, max_new_tokens=64,
        rep_penalty=5.0, rep_allowance=1, fusion_weight=0.5,
    )

    experiments = [
        ("baseline", None),
        ("horror",   ["horror nightmare monster ghost darkness fear terrifying"]),
        ("ocean",    ["ocean underwater coral reef fish diving deep sea"]),
    ]

    for name, target in experiments:
        print(f"\n{'─' * 60}")
        print(f"  {name.upper()}")
        print(f"{'─' * 60}")

        if target:
            sampler.set_guidance(target_texts=target, alpha=config.alpha)
            print(f"  Top tokens: {sampler.top_guided_tokens(8)}")
        else:
            sampler.clear_guidance()

        inputs = tokenizer.apply_chat_template(
            [prompt], add_generation_prompt=True, tokenize=True
        )
        if isinstance(inputs[0], int):
            inputs = [inputs]

        torch.manual_seed(42)
        outputs = sampler.sample(inputs, config, return_dict=True)

        for seq in outputs.sequences:
            response = clean_response(tokenizer.decode(seq, skip_special_tokens=False))
            print(f"\n  {response[:250]}")
            if target:
                metrics = evaluate_guidance(response, target, evaluator=sampler.embedder)
                print(f"  target_sim={metrics['target_sim']}, coh={metrics['coherence']}")


if __name__ == "__main__":
    main()
