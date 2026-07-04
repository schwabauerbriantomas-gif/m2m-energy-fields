"""
Phase 3 v7: EBM-Native Confidence Repair (Zero Extra Compute)

INSIGHT: The EBM energy field already IS a confidence landscape.
  token_scores[v] = dot(embed[v], d)
  - High positive → token is deeply aligned with target (confident)
  - Near zero → neutral (neither helps nor hurts)
  - Negative → anti-aligned (fights the energy field)

DSpark needs a separate confidence head because it has no energy model.
We don't — the energy score IS the confidence. Zero extra computation.

ALGORITHM:
  At each denoising step (standard MDLM loop):
    1. Forward pass → logits [B, T, V]        ← already doing this
    2. Energy guidance: logits += alpha * scores ← already doing this
    3. Commit tokens via topk confidence       ← already doing this
    4. REPAIR: for committed tokens where
       token_scores[committed_token] < energy_threshold
       → re-mask (set back to mask_token_id)
       Next step regenerates with updated context.
       ← NEW: but uses data already computed in step 2

COST: One gather + comparison per committed token. No extra forward pass.
"""

import os
import sys
import time
import json
import math

import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer

import dllm
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.core.samplers.utils import get_num_transfer_tokens, add_gumbel_noise
from dllm.utils import get_model, get_tokenizer

DEVICE = "cuda"
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "results", "phase3_v7_ebm_confidence_repair.jsonl")


def clean_response(text):
    if "<|start_header_id|>assistant<|end_header_id|>" in text:
        text = text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    text = text.replace("<|eot_id|>", "").replace("<|endoftext|>", "").strip()
    return text


def coherence_check(text, evaluator):
    words = text.split()
    if len(words) < 10:
        return 0.0
    mid = len(words) // 2
    h1, h2 = " ".join(words[:mid]), " ".join(words[mid:])
    e1 = evaluator.encode([h1], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
    e2 = evaluator.encode([h2], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
    return F.cosine_similarity(e1, e2).item()


def sample_with_ebm_repair(
    model, tokenizer, inputs, *,
    steps, max_new_tokens, block_size, temperature,
    target_texts, alpha, energy_threshold,
):
    """
    MDLM sampling loop with EBM energy guidance + native confidence repair.

    The repair uses token_scores (already computed for guidance) as confidence.
    Tokens committed with low energy alignment are re-masked for regeneration.
    """
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id
    embed_matrix = model.get_input_embeddings().weight.data.float()

    # ── Compute energy direction + token scores (done ONCE) ──
    embs = []
    for t in target_texts:
        tokens = tokenizer(t, return_tensors="pt", truncation=True, max_length=128)
        ids = tokens["input_ids"].to(DEVICE)
        with torch.no_grad():
            pooled = embed_matrix[ids].mean(dim=1).squeeze(0)
        embs.append(pooled)
    d = F.normalize(torch.stack(embs).mean(dim=0), dim=-1)
    with torch.no_grad():
        token_scores = torch.mv(embed_matrix, d)
        token_scores = token_scores / (token_scores.abs().max() + 1e-8)
        token_scores = token_scores.to(DEVICE)

    # ── Build canvas ──
    if isinstance(inputs[0], list):
        inputs = [torch.as_tensor(p, dtype=torch.long, device=DEVICE) for p in inputs]
    prompt_lens = [p.shape[0] for p in inputs]
    max_length = max_new_tokens + max(prompt_lens)
    B, T = len(inputs), max_length

    x = torch.full((B, T), eos_id, dtype=torch.long, device=DEVICE)
    for i, p in enumerate(inputs):
        x[i, :prompt_lens[i]] = p
        x[i, prompt_lens[i]:prompt_lens[i] + max_new_tokens] = mask_id

    attention_mask = torch.zeros((B, T), dtype=torch.long, device=DEVICE)
    for i, pl in enumerate(prompt_lens):
        attention_mask[i, :min(pl + max_new_tokens, T)] = 1

    # ── Block schedule ──
    from dllm.core.schedulers import LinearAlphaScheduler
    scheduler = LinearAlphaScheduler()
    num_blocks = math.ceil(max_new_tokens / block_size)
    steps_per_block = max(1, math.ceil(steps / num_blocks))

    repair_count = 0
    commit_count = 0
    histories = [x.clone()]

    for b in range(num_blocks):
        block_mask_index = torch.zeros((B, block_size), dtype=torch.bool, device=x.device)
        for j in range(B):
            start = prompt_lens[j] + b * block_size
            end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
            if start < end:
                block_mask_index[j, :end - start] = x[j, start:end] == mask_id

        num_transfer = get_num_transfer_tokens(
            mask_index=block_mask_index, steps=steps_per_block,
            scheduler=scheduler, stochastic=False,
        )
        effective_steps = num_transfer.size(1)

        for i in range(effective_steps):
            mask_index = x == mask_id

            # ── SINGLE forward pass ──
            with torch.no_grad():
                logits = model(x, attention_mask=attention_mask).logits

            # ── Energy guidance (modifies logits) ──
            scores = token_scores.unsqueeze(0).unsqueeze(0) * alpha
            mask = mask_index.unsqueeze(-1).float()
            logits = logits.float() + mask * scores

            # ── Token selection ──
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # Confidence for remasking (standard MDLM: use guided logits)
            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=x0.unsqueeze(-1)), -1)

            for j in range(B):
                x0_p[j, prompt_lens[j] + (b + 1) * block_size:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                k = int(num_transfer[j, i].item())
                if k > 0:
                    _, sel = torch.topk(confidence[j], k=k)
                    transfer_index[j, sel] = True

            # ── EBM NATIVE REPAIR ──
            # token_scores[x0] gives energy alignment of each committed token.
            # Already computed — zero extra cost.
            if energy_threshold > -1.0:  # -1 = disabled
                committed = transfer_index & mask_index
                # Energy confidence of committed tokens: token_scores[committed_token_id]
                committed_energy = token_scores[x0]  # [B, T]
                # Re-mask tokens below energy threshold
                low_energy = committed & (committed_energy < energy_threshold)
                transfer_index = transfer_index & ~low_energy
                repair_count += low_energy.sum().item()

            commit_count += transfer_index.sum().item()
            x[transfer_index] = x0[transfer_index]
            histories.append(x.clone())

    repair_rate = repair_count / max(commit_count, 1)
    return x, {
        "repairs": repair_count,
        "commits": commit_count,
        "repair_rate": round(repair_rate, 4),
    }


def run_experiment():
    print("=" * 70)
    print("Phase 3 v7: EBM-Native Confidence Repair (Zero Extra Compute)")
    print("=" * 70)

    print("\n[1] Loading LLaDA-8B...", flush=True)
    model = get_model(
        model_args=type("Args", (), {
            "model_name_or_path": MODEL_ID,
            "dtype": torch.bfloat16,
            "device_map": {"": 0},
        })()
    ).eval()
    tokenizer = get_tokenizer(model_args=type("Args", (), {"model_name_or_path": MODEL_ID})())
    evaluator = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
    print(f"  {torch.cuda.memory_allocated()/1e9:.2f} GB VRAM")

    prompt = [{"role": "user", "content": "Write a short story about something interesting."}]

    TOPICS = {
        "ocean":   ["ocean underwater coral reef fish diving deep sea submarine waves"],
        "horror":  ["horror nightmare monster ghost darkness fear terrifying scream blood"],
        "space":   ["space exploration stars Mars galaxies astronauts rocket launch mission"],
        "cooking": ["cooking recipe chef kitchen delicious food spices culinary restaurant"],
    }

    # ── Experiment design ──
    # For each topic: compare no-repair vs repair at different thresholds
    # energy_threshold: tokens with token_score < threshold get re-masked
    #   -1.0 = disabled (all tokens kept)
    #    0.0 = keep only energy-positive tokens
    #    0.1 = keep only well-aligned tokens
    #    0.3 = strict: only strongly aligned tokens survive

    experiments = [
        # Baseline (no guidance)
        {"name": "baseline",         "target": None,         "alpha": 0.0,  "eth": -1.0},
    ]

    for topic, target in TOPICS.items():
        experiments.append({"name": f"{topic}_a10_norepair",   "target": target, "alpha": 10.0, "eth": -1.0})
        experiments.append({"name": f"{topic}_a10_repair0",    "target": target, "alpha": 10.0, "eth":  0.0})
        experiments.append({"name": f"{topic}_a10_repair01",   "target": target, "alpha": 10.0, "eth":  0.1})
        experiments.append({"name": f"{topic}_a5_repair0",     "target": target, "alpha":  5.0, "eth":  0.0})
        experiments.append({"name": f"{topic}_a5_repair01",    "target": target, "alpha":  5.0, "eth":  0.1})

    results = []
    N = 2

    for exp in experiments:
        print(f"\n{'─' * 70}")
        print(f"  [{exp['name']}]")
        print(f"{'─' * 70}")

        inputs = tokenizer.apply_chat_template([prompt], add_generation_prompt=True, tokenize=True)
        if isinstance(inputs[0], int):
            inputs = [inputs]

        for trial in range(N):
            torch.manual_seed(42 + trial)

            if exp["alpha"] == 0:
                sampler = MDLMSampler(model=model, tokenizer=tokenizer)
                cfg = MDLMSamplerConfig(steps=128, max_new_tokens=128, block_size=32, temperature=0.6)
                outputs = sampler.sample(inputs, cfg, return_dict=True)
                sequences = outputs.sequences
                repair_stats = {"repairs": 0, "commits": 0, "repair_rate": 0}
            else:
                sequences, repair_stats = sample_with_ebm_repair(
                    model, tokenizer, inputs,
                    steps=128, max_new_tokens=128, block_size=32, temperature=0.6,
                    target_texts=exp["target"],
                    alpha=exp["alpha"],
                    energy_threshold=exp["eth"],
                )

            for seq in sequences:
                raw = tokenizer.decode(seq, skip_special_tokens=False)
                response = clean_response(raw)
                if len(response) < 15:
                    continue

                resp_emb = evaluator.encode([response], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
                metrics = {
                    "coherence": round(coherence_check(response, evaluator), 4),
                    "diversity": round(len(set(response.lower().split())) / max(len(response.split()), 1), 4),
                    "repair_rate": repair_stats["repair_rate"],
                }

                if exp["target"]:
                    target_embs = evaluator.encode(exp["target"], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
                    metrics["target_sim"] = round(F.cosine_similarity(resp_emb, target_embs).mean().item(), 4)

                result = {
                    "experiment": exp["name"],
                    "trial": trial,
                    "alpha": exp["alpha"],
                    "energy_threshold": exp["eth"],
                    "response": response,
                    **metrics,
                }
                results.append(result)

                sim_str = f"sim={metrics['target_sim']:.4f}" if "target_sim" in metrics else "sim=—"
                print(f"  [{trial}] {sim_str}, coh={metrics['coherence']:.4f}, div={metrics['diversity']:.4f}, repair={metrics['repair_rate']:.4f}", flush=True)
                print(f"       {response[:180]}", flush=True)

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY — Does EBM-native repair improve quality at high alpha?")
    print(f"{'=' * 70}")
    print(f"{'Experiment':<28} {'sim':>8} {'coh':>8} {'div':>8} {'repair':>8}")
    print("─" * 65)

    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        agg[r["experiment"]].append(r)

    for exp_name in [e["name"] for e in experiments]:
        trials = agg.get(exp_name, [])
        if not trials:
            continue
        sims = [t.get("target_sim", 0) for t in trials]
        cohs = [t["coherence"] for t in trials]
        divs = [t["diversity"] for t in trials]
        reps = [t["repair_rate"] for t in trials]
        has_sim = any("target_sim" in t for t in trials)
        sm = f"{np.mean(sims):.4f}" if has_sim else "   —"
        print(f"{exp_name:<28} {sm:>8} {np.mean(cohs):>8.4f} {np.mean(divs):>8.4f} {np.mean(reps):>8.4f}")

    # ── Key comparison: repair vs no-repair at same alpha ──
    print(f"\n{'=' * 70}")
    print("REPAIR EFFECT (does re-masking low-energy tokens help?)")
    print(f"{'=' * 70}")
    for topic in TOPICS:
        no_repair = [r for r in results if r["experiment"] == f"{topic}_a10_norepair"]
        repair0 = [r for r in results if r["experiment"] == f"{topic}_a10_repair0"]
        repair01 = [r for r in results if r["experiment"] == f"{topic}_a10_repair01"]

        for label, group in [("no_repair", no_repair), ("repair≥0", repair0), ("repair≥0.1", repair01)]:
            if group:
                cohs = [r["coherence"] for r in group]
                divs = [r["diversity"] for r in group]
                sims = [r.get("target_sim", 0) for r in group]
                print(f"  {topic:>10} {label:>12}: sim={np.mean(sims):.4f}, coh={np.mean(cohs):.4f}, div={np.mean(divs):.4f}")
        print()

    with open(RESULTS_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Results saved to {RESULTS_FILE}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_experiment()
