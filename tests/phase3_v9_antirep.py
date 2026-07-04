"""
Phase 3 v9: Energy Annealing + Anti-Repetition Logit Penalty

INSIGHT FROM v8b:
  Re-masking tokens doesn't work because the model proposes the same token
  again with the same energy active. And masks left unresolved at end of
  generation produce garbage output.

NEW APPROACH — Anti-Repetition Penalty (not repair):
  Instead of re-masking committed tokens, we PENALIZE their logits in
  subsequent steps. This is the frequency_penalty mechanism from
  autoregressive LMs, adapted for masked diffusion:

  After each commit step, we maintain a token frequency count in the
  generation region. For the next forward pass:
    logits[:, :, token] -= rep_penalty * max(0, count[token] - allowance)

  This means:
  - A token can appear 'allowance' times freely (e.g., 2)
  - After that, each additional occurrence gets harder to select
  - The penalty is cumulative — deeply repeated tokens get exponentially
    harder to commit

  ADVANTAGES over re-masking:
  1. No mask tokens left unresolved (we never create new masks)
  2. No deadlock (model picks the next-best token naturally)
  3. Zero extra forward passes (penalty applied to existing logits)
  4. Preserves the energy cascade (committed tokens still influence
     via bidirectional attention) but breaks the feedback loop

COMBINED WITH ENERGY ANNEALING:
  - Early steps: strong energy (topic steering) + weak penalty (let topic emerge)
  - Late steps: weak/no energy (coherence) + strong penalty (prevent repetition)
  - This naturally produces: topic set early, then model refines freely
"""

import os
import sys, time, json, math
import torch, torch.nn.functional as F, numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

import dllm
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.core.samplers.utils import get_num_transfer_tokens, add_gumbel_noise
from dllm.core.schedulers import LinearAlphaScheduler
from dllm.utils import get_model, get_tokenizer

DEVICE = "cuda"
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "results", "phase3_v9_antirep.jsonl")


def clean_response(text):
    if "<|start_header_id|>assistant<|end_header_id|>" in text:
        text = text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    return text.replace("<|eot_id|>", "").replace("<|endoftext|>", "").strip()


def coherence_check(text, evaluator):
    words = text.split()
    if len(words) < 10:
        return 0.0
    mid = len(words) // 2
    h1, h2 = " ".join(words[:mid]), " ".join(words[mid:])
    e1 = evaluator.encode([h1], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
    e2 = evaluator.encode([h2], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
    return F.cosine_similarity(e1, e2).item()


def sample_antirep(
    model, tokenizer, inputs, *,
    steps, max_new_tokens, block_size, temperature,
    target_texts,
    alpha_start, alpha_end, gamma,
    rep_penalty, rep_allowance,
):
    """
    MDLM sampling with:
      1. Energy annealing (alpha decays from alpha_start to alpha_end)
      2. Anti-repetition logit penalty (frequency_penalty style)

    rep_penalty: how much to subtract from logits per excess occurrence.
    rep_allowance: how many times a token can appear before penalty kicks in.
      allowance=2 → token freely appears twice, 3rd time gets -rep_penalty, etc.
    """
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id
    embed_matrix = model.get_input_embeddings().weight.data.float()

    # Energy direction + token scores
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
        token_scores = (token_scores / (token_scores.abs().max() + 1e-8)).to(DEVICE)

    # Canvas
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

    scheduler = LinearAlphaScheduler()
    num_blocks = math.ceil(max_new_tokens / block_size)
    steps_per_block = max(1, math.ceil(steps / num_blocks))
    total_steps = num_blocks * steps_per_block

    # Token frequency counter: [B, vocab] — how many times each token
    # has been committed in the generation region
    token_counts = torch.zeros(B, embed_matrix.shape[0], device=DEVICE)

    def alpha_at(step):
        progress = step / max(total_steps - 1, 1)
        decay = max(1.0 - progress, 0.0) ** gamma
        return alpha_start * decay + alpha_end * (1.0 - decay)

    # Penalty also anneals: weak early (let topic emerge), strong late
    def penalty_at(step):
        progress = step / max(total_steps - 1, 1)
        # Linear ramp from 0 to rep_penalty over the course of generation
        return rep_penalty * progress

    global_step = 0

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
            a = alpha_at(global_step)
            pen = penalty_at(global_step)

            # ── Forward pass ──
            with torch.no_grad():
                logits = model(x, attention_mask=attention_mask).logits.float()

            # ── Energy guidance ──
            energy_scores = token_scores.unsqueeze(0).unsqueeze(0) * a
            logits = logits + mask_index.unsqueeze(-1).float() * energy_scores

            # ── Anti-repetition penalty ──
            # For each token v: if count[v] > allowance, subtract penalty
            # Penalty scales with excess count and ramps up over generation
            if pen > 0 and global_step > 2:
                excess = (token_counts - rep_allowance).clamp(min=0)  # [B, vocab]
                # penalty[b, v] = pen * excess[b, v]
                penalty_matrix = pen * excess  # [B, vocab]
                # Apply to all positions: logits[b, :, v] -= penalty_matrix[b, v]
                logits = logits - penalty_matrix.unsqueeze(1)  # broadcast [B, 1, vocab]

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

            # ── Commit + update frequency counts ──
            committed_tokens = x0[transfer_index]  # [num_committed]
            for j in range(B):
                mask_j = transfer_index[j]
                committed_j = x0[j][mask_j]
                token_counts[j].scatter_add_(
                    0, committed_j,
                    torch.ones_like(committed_j, dtype=token_counts.dtype),
                )

            x[transfer_index] = x0[transfer_index]
            global_step += 1

    return x


def run_experiment():
    print("=" * 70)
    print("Phase 3 v9: Energy Annealing + Anti-Repetition Penalty")
    print("=" * 70)

    model = get_model(model_args=type("Args", (), {
        "model_name_or_path": MODEL_ID, "dtype": torch.bfloat16, "device_map": {"": 0},
    })()).eval()
    tokenizer = get_tokenizer(model_args=type("Args", (), {"model_name_or_path": MODEL_ID})())
    evaluator = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
    print(f"  {torch.cuda.memory_allocated()/1e9:.2f} GB VRAM\n")

    prompt = [{"role": "user", "content": "Write a short story about something interesting."}]

    TOPICS = {
        "ocean":   ["ocean underwater coral reef fish diving deep sea submarine waves"],
        "horror":  ["horror nightmare monster ghost darkness fear terrifying scream blood"],
        "space":   ["space exploration stars Mars galaxies astronauts rocket launch mission"],
        "cooking": ["cooking recipe chef kitchen delicious food spices culinary restaurant"],
    }

    # ── Configs ──
    # penalty=0 → pure annealing (v8b reference)
    # penalty=2, allowance=2 → mild: token can appear twice, then -2 per extra
    # penalty=5, allowance=2 → strong: token can appear twice, then -5 per extra
    # penalty=5, allowance=1 → very strong: token can appear once, then -5

    experiments = []

    # Baseline (no guidance, no penalty)
    experiments.append(("baseline", None, 0, 0, 1, 0, 0))

    for topic, target in TOPICS.items():
        # Reference: constant alpha=5, no penalty
        experiments.append((f"{topic}_const5",       target, 5,  5,  1, 0, 0))
        # Anneal without penalty (v8b reference)
        experiments.append((f"{topic}_anneal_noprob", target, 10, 0,  1, 0, 0))
        # Anneal + mild penalty
        experiments.append((f"{topic}_anneal_p2",    target, 10, 0,  1, 2, 2))
        # Anneal + strong penalty
        experiments.append((f"{topic}_anneal_p5",    target, 10, 0,  1, 5, 2))
        # Anneal + strong penalty, allowance=1
        experiments.append((f"{topic}_anneal_p5a1",  target, 10, 0,  1, 5, 1))

    results = []

    for name, target, a_start, a_end, gamma, pen, allow in experiments:
        print(f"[{name}]")

        inputs = tokenizer.apply_chat_template([prompt], add_generation_prompt=True, tokenize=True)
        if isinstance(inputs[0], int):
            inputs = [inputs]

        torch.manual_seed(42)

        if a_start == 0:
            # Baseline: standard MDLM
            sampler = MDLMSampler(model=model, tokenizer=tokenizer)
            cfg = MDLMSamplerConfig(steps=64, max_new_tokens=64, block_size=32, temperature=0.6)
            outputs = sampler.sample(inputs, cfg, return_dict=True)
            sequences = outputs.sequences
        else:
            sequences = sample_antirep(
                model, tokenizer, inputs,
                steps=64, max_new_tokens=64, block_size=32, temperature=0.6,
                target_texts=target,
                alpha_start=a_start, alpha_end=a_end, gamma=gamma,
                rep_penalty=pen, rep_allowance=allow,
            )

        for seq in sequences:
            response = clean_response(tokenizer.decode(seq, skip_special_tokens=False))
            if len(response) < 15:
                continue

            resp_emb = evaluator.encode([response], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
            metrics = {
                "coherence": round(coherence_check(response, evaluator), 4),
                "diversity": round(len(set(response.lower().split())) / max(len(response.split()), 1), 4),
            }

            if target:
                target_embs = evaluator.encode(target, convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
                metrics["target_sim"] = round(F.cosine_similarity(resp_emb, target_embs).mean().item(), 4)

            s = metrics.get("target_sim", 0)
            metrics["quality"] = round(s * metrics["coherence"] * metrics["diversity"], 4)

            results.append({
                "experiment": name, "target": target is not None,
                "alpha": a_start, "penalty": pen, "allowance": allow,
                "response": response, **metrics,
            })

            verdict = "✅" if metrics["quality"] > 0.05 else ("⚠️" if metrics["quality"] > 0.01 else "❌")
            sim_str = f"sim={metrics.get('target_sim', '—')}"
            print(f"  {verdict} {sim_str} coh={metrics['coherence']:.4f} div={metrics['diversity']:.4f} q={metrics['quality']:.4f}")
            print(f"    {response[:160]}\n")

    # ── Summary ──
    print(f"\n{'=' * 75}")
    print("ANTI-REP PENALY COMPARISON")
    print(f"{'=' * 75}")
    print(f"{'Config':<28} {'sim':>8} {'coh':>8} {'div':>8} {'quality':>8}")
    print("─" * 75)

    agg = defaultdict(list)
    for r in results:
        agg[r["experiment"]].append(r)

    for name in [e[0] for e in experiments]:
        trials = agg.get(name, [])
        if not trials:
            continue
        sims = [t.get("target_sim", 0) for t in trials]
        cohs = [t["coherence"] for t in trials]
        divs = [t["diversity"] for t in trials]
        quals = [t["quality"] for t in trials]
        has_sim = any("target_sim" in t for t in trials)
        sm = f"{np.mean(sims):.4f}" if has_sim else "   —"
        print(f"{name:<28} {sm:>8} {np.mean(cohs):>8.4f} {np.mean(divs):>8.4f} {np.mean(quals):>8.4f}")

    # Per-topic comparison
    print(f"\n{'=' * 75}")
    print("BY TOPIC: no-penalty vs penalty configs")
    print(f"{'=' * 75}")
    for topic in TOPICS:
        print(f"\n  {topic.upper()}:")
        for suffix, label in [
            ("_const5",       "const α=5"),
            ("_anneal_noprob", "anneal 10→0, no penalty"),
            ("_anneal_p2",    "anneal 10→0, penalty=2 allow=2"),
            ("_anneal_p5",    "anneal 10→0, penalty=5 allow=2"),
            ("_anneal_p5a1",  "anneal 10→0, penalty=5 allow=1"),
        ]:
            name = f"{topic}{suffix}"
            trials = agg.get(name, [])
            if not trials:
                continue
            t = trials[0]
            print(f"    {label:<38}  sim={t.get('target_sim', 0):.4f}  coh={t['coherence']:.4f}  div={t['diversity']:.4f}  q={t['quality']:.4f}")

    # Best outputs
    print(f"\n{'=' * 75}")
    print("TOP 8 OUTPUTS (by quality)")
    print(f"{'=' * 75}")
    for r in sorted(results, key=lambda x: x["quality"], reverse=True)[:8]:
        print(f"\n  [{r['experiment']}] q={r['quality']:.4f}")
        print(f"  sim={r.get('target_sim', 0):.4f} coh={r['coherence']:.4f} div={r['diversity']:.4f}")
        print(f"  {r['response'][:250]}")

    with open(RESULTS_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    run_experiment()
