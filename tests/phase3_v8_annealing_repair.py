"""
Phase 3 v8: Energy Annealing + Repetition-Aware Repair

ROOT CAUSE of repetition collapse:
  At high alpha, energy forces the same high-score token at every position.
  The model's bidirectional attention sees the repeated token and reinforces it.
  This creates a positive feedback loop: "scary scary scary scary..."

SOLUTION — two mechanisms working together:

  1. ENERGY ANNEALING:
     Alpha decays across denoising steps. Strong early (topic steering),
     weak late (coherence refinement).
       alpha(step) = alpha_start * (1 - step/total_steps)^gamma
     This lets energy set the topic in the first ~30% of steps, then
     hands control to the model for grammar/coherence.

  2. REPETITION-AWARE REPAIR:
     After committing tokens, detect consecutive repetition (same token
     3+ times in a row). Re-mask the repeated tokens. On regeneration:
       - alpha for those positions = 0 (no energy forcing)
       - Model uses its own logits with full bidirectional context
     This breaks the feedback loop WITHOUT deadlocking, because we
     only target actual repetition, not all low-energy tokens.

ZERO EXTRA COMPUTE:
  - Annealing: just multiply alpha by a scalar per step
  - Repetition detection: tensor comparison on committed tokens
  - No extra forward passes needed

DSpark analogy:
  DSpark prunes low-confidence draft tokens and regenerates with the
  target model (no draft influence). We do the same: prune repetitive
  tokens and regenerate with the model (no energy influence on those
  positions).
"""

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
from dllm.core.schedulers import LinearAlphaScheduler
from dllm.utils import get_model, get_tokenizer

DEVICE = "cuda"
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
RESULTS_FILE = "/root/m2m-energy-fields/results/phase3_v8_annealing_repair.jsonl"


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


def detect_repetition(x, mask_id, prompt_lens, rep_threshold=3):
    """
    Find positions where the same token appears >= rep_threshold times
    consecutively. Returns boolean mask [B, T] of positions to re-mask.

    Only checks the generation region (after prompt).
    """
    B, T = x.shape
    repair_mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)

    for j in range(B):
        gen_start = prompt_lens[j]
        gen_tokens = x[j, gen_start:]

        # Find runs of identical tokens (excluding mask and special tokens)
        for i in range(len(gen_tokens)):
            tid = gen_tokens[i].item()
            if tid == mask_id:
                continue

            # Count forward run length
            run_len = 1
            for k in range(i + 1, min(i + 20, len(gen_tokens))):
                if gen_tokens[k].item() == tid:
                    run_len += 1
                else:
                    break

            if run_len >= rep_threshold:
                # Re-mask all tokens in this run except the first one
                # Keep the first occurrence (it's legitimate context)
                for k in range(i + 1, i + run_len):
                    if gen_tokens[k].item() == tid:
                        repair_mask[j, gen_start + k] = True

    return repair_mask


def sample_annealed(
    model, tokenizer, inputs, *,
    steps, max_new_tokens, block_size, temperature,
    target_texts, alpha_start, alpha_end, gamma,
    rep_threshold, rep_repair_ratio,
):
    """
    MDLM sampling with energy annealing + repetition-aware repair.

    Args:
        alpha_start: Energy strength at step 0
        alpha_end: Energy strength at final step (typically 0)
        gamma: Annealing curve. gamma=1 → linear, gamma=2 → quadratic (slow start)
        rep_threshold: Re-mask runs of >= this many identical tokens
        rep_repair_ratio: Fraction of repetition to repair (0-1).
            1.0 = repair all, 0.5 = repair half (stochastic)
    """
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id
    embed_matrix = model.get_input_embeddings().weight.data.float()

    # ── Energy direction + token scores ──
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
    scheduler = LinearAlphaScheduler()
    num_blocks = math.ceil(max_new_tokens / block_size)
    steps_per_block = max(1, math.ceil(steps / num_blocks))
    total_effective_steps = num_blocks * steps_per_block

    # ── Per-step alpha schedule ──
    # alpha(step) = alpha_start * (1 - step/total)^gamma
    # But clamp to alpha_end minimum
    step_alphas = []
    for s in range(total_effective_steps):
        progress = s / max(total_effective_steps - 1, 1)
        decay = max(1.0 - progress, 0.0) ** gamma
        a = alpha_start * decay + alpha_end * (1.0 - decay)
        step_alphas.append(a)

    global_step = 0
    total_repairs = 0
    total_commits = 0

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
            current_alpha = step_alphas[min(global_step, len(step_alphas) - 1)]

            # ── Forward pass ──
            with torch.no_grad():
                logits = model(x, attention_mask=attention_mask).logits

            # ── Energy guidance with ANNEALED alpha ──
            scores = token_scores.unsqueeze(0).unsqueeze(0) * current_alpha
            mask = mask_index.unsqueeze(-1).float()
            logits = logits.float() + mask * scores

            # ── Token selection ──
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

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

            # ── Commit tokens ──
            x[transfer_index] = x0[transfer_index]
            total_commits += transfer_index.sum().item()

            # ── REPETITION-AWARE REPAIR ──
            if rep_threshold > 0 and global_step > 2:
                rep_mask = detect_repetition(
                    x, mask_id, prompt_lens, rep_threshold
                )
                if rep_repair_ratio < 1.0:
                    # Stochastic repair: only repair a fraction
                    rand = torch.rand_like(rep_mask, dtype=torch.float)
                    rep_mask = rep_mask & (rand < rep_repair_ratio)

                num_rep = rep_mask.sum().item()
                if num_rep > 0:
                    # Re-mask the repeated tokens
                    x[rep_mask] = mask_id
                    total_repairs += num_rep

            global_step += 1

    repair_rate = total_repairs / max(total_commits, 1)
    return x, {
        "repairs": total_repairs,
        "commits": total_commits,
        "repair_rate": round(repair_rate, 4),
    }


def run_experiment():
    print("=" * 70)
    print("Phase 3 v8: Energy Annealing + Repetition-Aware Repair")
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

    # ── Experiment configs ──
    # Compare: constant alpha (v7 baseline) vs annealed alpha vs annealed+repair
    experiments = [
        # Baseline: no guidance
        {"name": "baseline",            "target": None, "a_start": 0,   "a_end": 0,   "gamma": 1, "rep_thresh": 0, "rep_ratio": 0},
    ]

    for topic, target in TOPICS.items():
        # v7 reference: constant alpha=10 (known to collapse)
        experiments.append({"name": f"{topic}_const10",    "target": target, "a_start": 10,  "a_end": 10,  "gamma": 1, "rep_thresh": 0, "rep_ratio": 0})
        # v7 reference: constant alpha=5 (best from v4)
        experiments.append({"name": f"{topic}_const5",     "target": target, "a_start": 5,   "a_end": 5,   "gamma": 1, "rep_thresh": 0, "rep_ratio": 0})

        # ANNEALING: start strong, decay to 0
        experiments.append({"name": f"{topic}_anneal10_0", "target": target, "a_start": 10,  "a_end": 0,   "gamma": 1, "rep_thresh": 0, "rep_ratio": 0})
        experiments.append({"name": f"{topic}_anneal15_0", "target": target, "a_start": 15,  "a_end": 0,   "gamma": 2, "rep_thresh": 0, "rep_ratio": 0})

        # ANNEALING + REPETITION REPAIR
        experiments.append({"name": f"{topic}_anneal10_rep",  "target": target, "a_start": 10,  "a_end": 0,   "gamma": 1, "rep_thresh": 3, "rep_ratio": 1.0})
        experiments.append({"name": f"{topic}_anneal15_rep",  "target": target, "a_start": 15,  "a_end": 0,   "gamma": 2, "rep_thresh": 3, "rep_ratio": 1.0})

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

            if exp["a_start"] == 0:
                # Baseline: standard MDLM
                sampler = MDLMSampler(model=model, tokenizer=tokenizer)
                cfg = MDLMSamplerConfig(steps=128, max_new_tokens=128, block_size=32, temperature=0.6)
                outputs = sampler.sample(inputs, cfg, return_dict=True)
                sequences = outputs.sequences
                repair_stats = {"repairs": 0, "commits": 0, "repair_rate": 0}
            else:
                sequences, repair_stats = sample_annealed(
                    model, tokenizer, inputs,
                    steps=128, max_new_tokens=128, block_size=32, temperature=0.6,
                    target_texts=exp["target"],
                    alpha_start=exp["a_start"],
                    alpha_end=exp["a_end"],
                    gamma=exp["gamma"],
                    rep_threshold=exp["rep_thresh"],
                    rep_repair_ratio=exp["rep_ratio"],
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

                # Quality score: harmonic mean of sim × coh × div
                s = metrics.get("target_sim", 0)
                c = metrics["coherence"]
                d = metrics["diversity"]
                metrics["quality"] = round(s * c * d, 4)

                result = {
                    "experiment": exp["name"],
                    "trial": trial,
                    "a_start": exp["a_start"],
                    "a_end": exp["a_end"],
                    "gamma": exp["gamma"],
                    "rep_thresh": exp["rep_thresh"],
                    "response": response,
                    **metrics,
                }
                results.append(result)

                sim_str = f"sim={metrics.get('target_sim', '—')}"
                print(f"  [{trial}] {sim_str}, coh={c:.4f}, div={d:.4f}, q={metrics['quality']:.4f}, repair={metrics['repair_rate']:.4f}", flush=True)
                print(f"       {response[:180]}", flush=True)

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Experiment':<28} {'sim':>8} {'coh':>8} {'div':>8} {'quality':>8} {'repair':>8}")
    print("─" * 70)

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
        quals = [t["quality"] for t in trials]
        reps = [t["repair_rate"] for t in trials]
        has_sim = any("target_sim" in t for t in trials)
        sm = f"{np.mean(sims):.4f}" if has_sim else "   —"
        print(f"{exp_name:<28} {sm:>8} {np.mean(cohs):>8.4f} {np.mean(divs):>8.4f} {np.mean(quals):>8.4f} {np.mean(reps):>8.4f}")

    # ── Key comparison ──
    print(f"\n{'=' * 70}")
    print("CONSTANT vs ANNEALED vs ANNEALED+REPAIR (by topic)")
    print(f"{'=' * 70}")
    for topic in TOPICS:
        print(f"\n  {topic.upper()}:")
        for suffix, label in [
            ("_const10",    "constant α=10"),
            ("_const5",     "constant α=5"),
            ("_anneal10_0", "anneal 10→0"),
            ("_anneal15_0", "anneal 15→0 (γ=2)"),
            ("_anneal10_rep", "anneal 10→0 + repair"),
            ("_anneal15_rep", "anneal 15→0 + repair (γ=2)"),
        ]:
            name = f"{topic}{suffix}"
            trials = agg.get(name, [])
            if not trials:
                continue
            sims = [t.get("target_sim", 0) for t in trials]
            cohs = [t["coherence"] for t in trials]
            divs = [t["diversity"] for t in trials]
            quals = [t["quality"] for t in trials]
            best_q = max(quals)
            print(f"    {label:<26}  sim={np.mean(sims):.4f}  coh={np.mean(cohs):.4f}  div={np.mean(divs):.4f}  q_best={best_q:.4f}")

    # ── Best examples ──
    print(f"\n{'=' * 70}")
    print("BEST EXAMPLES (highest quality = sim × coh × div)")
    print(f"{'=' * 70}")
    guided = [r for r in results if "target_sim" in r and r["target_sim"] > 0]
    guided.sort(key=lambda x: x["quality"], reverse=True)
    for r in guided[:8]:
        print(f"\n  [{r['experiment']}] q={r['quality']:.4f} (sim={r['target_sim']:.4f}, coh={r['coherence']:.4f}, div={r['diversity']:.4f})")
        print(f"  {r['response'][:250]}")

    with open(RESULTS_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_experiment()
