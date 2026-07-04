"""
Example: Energy-guided story generation with LLaDA-8B.

This script demonstrates the core capability of m2m-energy-fields:
steering a masked diffusion model toward a topic using energy fields,
without mentioning the topic in the prompt.

Requirements:
    - LLaDA-8B-Instruct downloaded (16GB VRAM)
    - dllm framework installed
    - m2m-energy-fields installed (pip install -e .)

Usage:
    python examples/guided_story_llada8b.py
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from m2m_energy_fields import EnergyGuidedSampler, GuidanceConfig
from m2m_energy_fields.metrics import evaluate_guidance
from dllm.utils import get_model, get_tokenizer

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"


def clean_response(text: str) -> str:
    """Extract assistant response from chat template."""
    if "<|start_header_id|>assistant<|end_header_id|>" in text:
        text = text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    text = text.replace("<|eot_id|>", "").replace("<|endoftext|>", "").strip()
    return text


def main():
    print("=" * 60)
    print("m2m-energy-fields: Guided Story Generation")
    print("=" * 60)

    # ── Load model ──
    print("\n[1] Loading LLaDA-8B-Instruct...", flush=True)
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
    print(f"  {torch.cuda.memory_allocated()/1e9:.2f} GB VRAM")

    # ── Create sampler ──
    print("\n[2] Creating EnergyGuidedSampler (building MiniLM table)...", flush=True)
    sampler = EnergyGuidedSampler(model=model, tokenizer=tokenizer)

    # ── Open prompt (topic is NOT in the prompt) ──
    prompt = [{"role": "user", "content": "Write a short story about something interesting."}]

    config = GuidanceConfig(
        alpha=10.0,      # strong initial steering
        temperature=0.6,
        steps=64,
        max_new_tokens=64,
        rep_penalty=5.0,
        rep_allowance=1,
        fusion_weight=0.5,  # convex 50/50
    )

    experiments = [
        ("baseline", None),
        ("ocean",    ["ocean underwater coral reef fish diving deep sea"]),
        ("space",    ["space exploration stars Mars galaxies astronauts"]),
        ("horror",   ["horror nightmare monster ghost darkness fear terrifying"]),
        ("cooking",  ["cooking recipe chef kitchen delicious food"]),
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
            print(f"\n  Response:")
            print(f"  {response[:300]}")
            if target:
                metrics = evaluate_guidance(response, target, evaluator=sampler.embedder)
                print(f"\n  target_sim: {metrics['target_sim']}")
                print(f"  coherence:  {metrics['coherence']}")
                print(f"  diversity:  {metrics['diversity']}")


if __name__ == "__main__":
    main()
