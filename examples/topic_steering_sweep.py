"""
Reproducible experiment: topic steering sweep.

Runs the full experiment described in the README — generates stories with
different energy targets and alpha values, evaluates target_sim / coherence /
diversity, and saves results to JSONL.

Usage:
    python examples/topic_steering_sweep.py [--model llada8b|qwen3]

Results are saved to results/topic_steering_<model>.jsonl
"""

import sys
import os
import json
import time
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from m2m_energy_fields import EnergyGuidedSampler, GuidanceConfig
from m2m_energy_fields.metrics import evaluate_guidance
from dllm.utils import get_model, get_tokenizer


MODELS = {
    "llada8b": "GSAI-ML/LLaDA-8B-Instruct",
    "qwen3": "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
}


def clean_response(text: str, model_key: str) -> str:
    if model_key == "qwen3":
        if "<|im_start|>assistant" in text:
            text = text.split("<|im_start|>assistant")[-1]
        if "<think>" in text and "</think>" in text:
            s, e = text.find("<think>"), text.find("</think>") + len("</think>")
            text = text[:s] + text[e:]
        text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    else:
        if "<|start_header_id|>assistant<|end_header_id|>" in text:
            text = text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        text = text.replace("<|eot_id|>", "").replace("<|endoftext|>", "").strip()
    return text


TOPICS = {
    "space":   ["space exploration stars Mars galaxies astronauts rocket launch mission"],
    "ocean":   ["ocean underwater coral reef fish diving deep sea submarine waves"],
    "horror":  ["horror nightmare monster ghost darkness fear terrifying scream blood"],
    "cooking": ["cooking recipe chef kitchen delicious food spices culinary restaurant"],
}


def main():
    parser = argparse.ArgumentParser(description="Topic steering experiment")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="llada8b")
    parser.add_argument("--n-samples", type=int, default=3)
    args = parser.parse_args()

    model_id = MODELS[args.model]
    print(f"Model: {model_id}", flush=True)

    # Load
    model = get_model(
        model_args=type("Args", (), {
            "model_name_or_path": model_id,
            "dtype": torch.bfloat16,
            "device_map": {"": 0},
        })()
    ).eval()
    tokenizer = get_tokenizer(model_args=type("Args", (), {"model_name_or_path": model_id})())
    sampler = EnergyGuidedSampler(model=model, tokenizer=tokenizer)

    prompt = [{"role": "user", "content": "Write a short story about something interesting."}]
    results = []

    config = GuidanceConfig(
        alpha=10.0, temperature=0.6, steps=64, max_new_tokens=64,
        rep_penalty=5.0, rep_allowance=1, fusion_weight=0.5,
    )

    # Baseline first
    print("\n[baseline]", flush=True)
    sampler.clear_guidance()
    inputs = tokenizer.apply_chat_template([prompt], add_generation_prompt=True, tokenize=True)
    if isinstance(inputs[0], int):
        inputs = [inputs]
    torch.manual_seed(42)
    outputs = sampler.sample(inputs, config, return_dict=True)
    for seq in outputs.sequences:
        response = clean_response(tokenizer.decode(seq, skip_special_tokens=False), args.model)
        metrics = evaluate_guidance(response, evaluator=sampler.embedder)
        results.append({"experiment": "baseline", "alpha": 0.0, "trial": 0,
                        "response": response, **metrics})
        print(f"  coh={metrics['coherence']:.4f} div={metrics['diversity']:.4f}", flush=True)

    # Guided
    for topic_name, target in TOPICS.items():
        print(f"\n[{topic_name}]", flush=True)
        sampler.set_guidance(target_texts=target, alpha=config.alpha)
        print(f"  Top tokens: {sampler.top_guided_tokens(6)}", flush=True)

        for trial in range(args.n_samples):
            torch.manual_seed(42 + trial)
            inputs = tokenizer.apply_chat_template(
                [prompt], add_generation_prompt=True, tokenize=True
            )
            if isinstance(inputs[0], int):
                inputs = [inputs]

            outputs = sampler.sample(inputs, config, return_dict=True)

            for seq in outputs.sequences:
                response = clean_response(
                    tokenizer.decode(seq, skip_special_tokens=False), args.model
                )
                if len(response) < 15:
                    continue
                metrics = evaluate_guidance(response, target, evaluator=sampler.embedder)
                quality = metrics.get("target_sim", 0) * metrics["coherence"] * metrics["diversity"]
                result = {
                    "experiment": f"{topic_name}_convex50",
                    "topic": topic_name,
                    "alpha": config.alpha,
                    "trial": trial,
                    "response": response,
                    "quality": round(quality, 4),
                    **metrics,
                }
                results.append(result)
                print(f"  [{trial}] sim={metrics.get('target_sim', '—')}, "
                      f"coh={metrics['coherence']:.4f}, q={quality:.4f}", flush=True)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    agg = defaultdict(list)
    for r in results:
        agg[r["experiment"]].append(r)

    print(f"{'Experiment':<25} {'sim':>8} {'coh':>8} {'div':>8} {'quality':>8}")
    print("─" * 60)
    for exp_name in sorted(agg.keys()):
        trials = agg[exp_name]
        sims = [t.get("target_sim", 0) for t in trials]
        cohs = [t["coherence"] for t in trials]
        divs = [t["diversity"] for t in trials]
        quals = [t.get("quality", 0) for t in trials]
        has_sim = any("target_sim" in t for t in trials)
        sm = f"{np.mean(sims):.4f}" if has_sim else "   —"
        print(f"{exp_name:<25} {sm:>8} {np.mean(cohs):>8.4f} {np.mean(divs):>8.4f} {np.mean(quals):>8.4f}")

    out_file = os.path.join(os.path.dirname(__file__), "..", "results", f"topic_steering_{args.model}.jsonl")
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
