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
import json
import time
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer

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
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 2.0, 5.0, 10.0])
    args = parser.parse_args()

    model_id = MODELS[args.model]
    print(f"Model: {model_id}", flush=True)
    print(f"Alphas: {args.alphas}", flush=True)

    # Load
    model = get_model(
        model_args=type("Args", (), {
            "model_name_or_path": model_id,
            "dtype": torch.bfloat16,
            "device_map": {"": 0},
        })()
    ).eval()
    tokenizer = get_tokenizer(model_args=type("Args", (), {"model_name_or_path": model_id})())
    evaluator = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    sampler = EnergyGuidedSampler(model=model, tokenizer=tokenizer)

    prompt = [{"role": "user", "content": "Write a short story about something interesting."}]
    results = []

    for topic_name, target in TOPICS.items():
        for alpha in args.alphas:
            label = f"{topic_name}_a{alpha}" if alpha > 0 else "baseline"
            print(f"\n[{label}]", flush=True)

            sampler.set_guidance(
                target_texts=target if alpha > 0 else None,
                alpha=alpha,
            )
            if alpha > 0:
                print(f"  Top tokens: {sampler.top_guided_tokens(6)}", flush=True)

            config = GuidanceConfig(alpha=alpha, temperature=0.6)

            for trial in range(args.n_samples):
                torch.manual_seed(42 + trial)
                inputs = tokenizer.apply_chat_template(
                    [prompt], add_generation_prompt=True, tokenize=True
                )
                if isinstance(inputs[0], int):
                    inputs = [inputs]

                with sampler:
                    outputs = sampler.sample(inputs, config)

                for seq in outputs.sequences:
                    response = clean_response(
                        tokenizer.decode(seq, skip_special_tokens=False), args.model
                    )
                    if len(response) < 15:
                        continue
                    metrics = evaluate_guidance(response, target, evaluator=evaluator)
                    result = {
                        "experiment": label,
                        "topic": topic_name,
                        "alpha": alpha,
                        "trial": trial,
                        "response": response,
                        **metrics,
                    }
                    results.append(result)
                    print(f"  [{trial}] sim={metrics.get('target_sim', '—')}, coh={metrics['coherence']}", flush=True)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        agg[r["experiment"]].append(r)

    print(f"{'Experiment':<22} {'alpha':>6} {'sim':>8} {'coh':>8} {'div':>8}")
    print("─" * 55)
    for exp_name in sorted(agg.keys()):
        trials = agg[exp_name]
        sims = [t.get("target_sim", 0) for t in trials]
        cohs = [t.get("coherence", 0) for t in trials]
        divs = [t.get("diversity", 0) for t in trials]
        has_sim = any("target_sim" in t for t in trials)
        sm = f"{np.mean(sims):.4f}" if has_sim else "   —"
        print(f"{exp_name:<22} {trials[0]['alpha']:>6.1f} {sm:>8} {np.mean(cohs):>8.4f} {np.mean(divs):>8.4f}")

    out_file = f"results/topic_steering_{args.model}.jsonl"
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
