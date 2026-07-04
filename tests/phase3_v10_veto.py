"""
Phase 3 v10: Energy-Model Agreement Veto (DSpark propose-verify pattern)

DSPARK PATTERN:
  Draft model proposes tokens → confidence head evaluates → target model
  accepts or rejects based on agreement.

OUR TRANSLATION:
  Energy field proposes tokens (via guided logits) → model evaluates (via
  original logits) → if model disagrees strongly enough, VETO the energy
  proposal and commit the model's own preference instead.

MECHANISM (zero extra forward pass):
  At each denoising step we already compute:
    guided_logits = original_logits + alpha * mask * energy_scores
    x0_guided    = argmax(guided_logits)     ← energy's choice
    model_probs   = softmax(original_logits)  ← model's opinion
    p_agreement   = model_probs[x0_guided]    ← how much model agrees

  VETO RULE:
    if p_agreement < veto_threshold:
        # Model strongly disagrees with energy's choice
        # Commit model's own argmax instead
        x0_final = argmax(original_logits)
    else:
        # Model agrees (or doesn't care strongly)
        # Commit energy's choice
        x0_final = x0_guided

WHY THIS BEATS ANTI-REP PENALTY (v9):
  - v9 counts repetitions after they happen and penalizes retroactively
  - v10 prevents the repetition from entering the canvas in the first place
  - v10 is self-adaptive: as tokens accumulate, bidirectional attention
    naturally lowers model_prob for repeated tokens → veto triggers
    automatically without any counter
  - v10 preserves energy steering when the model is neutral or agreeable
  - v10 has zero state: no token_counts tensor, no scatter_add

COMBINED WITH ANNEALING:
  Early steps: alpha high, veto_threshold moderate → energy steers hard
    but model can override egregious choices
  Late steps: alpha ~0, veto_threshold irrelevant → model controls freely
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
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "results", "phase3_v10_veto.jsonl")


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


def sample_veto(
    model, tokenizer, inputs, *,
    steps, max_new_tokens, block_size, temperature,
    target_texts,
    alpha_start, alpha_end, gamma,
    veto_threshold,
):
    """
    MDLM sampling with energy annealing + model-agreement veto.

    veto_threshold: if model's softmax probability for energy's chosen token
    is below this, veto the energy choice and use model's own argmax.
    - 0.0 = never veto (pure energy guidance)
    - 0.1 = veto only when model strongly disagrees
    - 0.3 = moderate veto
    - 0.5 = aggressive veto (model gets final say on most tokens)
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

    def alpha_at(step):
        progress = step / max(total_steps - 1, 1)
        decay = max(1.0 - progress, 0.0) ** gamma
        return alpha_start * decay + alpha_end * (1.0 - decay)

    global_step = 0
    veto_count = 0
    commit_count = 0

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

            # ── SINGLE forward pass ──
            with torch.no_grad():
                raw_logits = model(x, attention_mask=attention_mask).logits.float()

            # Keep original logits for veto decision
            model_logits = raw_logits.clone()

            # Apply energy guidance
            energy_scores = token_scores.unsqueeze(0).unsqueeze(0) * a
            guided_logits = raw_logits + mask_index.unsqueeze(-1).float() * energy_scores

            # ── Energy proposes ──
            guided_noise = add_gumbel_noise(guided_logits, temperature=temperature)
            x0_energy = torch.argmax(guided_noise, dim=-1)

            # ── Model evaluates ──
            model_probs = F.softmax(model_logits, dim=-1)
            p_agree = torch.squeeze(
                torch.gather(model_probs, dim=-1, index=x0_energy.unsqueeze(-1)), -1
            )

            # ── VETO: where model strongly disagrees, use model's own choice ──
            disagree = (p_agree < veto_threshold) & mask_index
            if disagree.any():
                model_noise = add_gumbel_noise(model_logits, temperature=temperature)
                x0_model = torch.argmax(model_noise, dim=-1)
                x0 = torch.where(disagree, x0_model, x0_energy)
                veto_count += disagree.sum().item()
            else:
                x0 = x0_energy

            # Confidence for remasking (use guided logits for ranking)
            p_guided = F.softmax(guided_logits, dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p_guided, dim=-1, index=x0.unsqueeze(-1)), -1
            )

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

            x[transfer_index] = x0[transfer_index]
            commit_count += transfer_index.sum().item()
            global_step += 1

    veto_rate = veto_count / max(commit_count, 1)
    return x, {"vetoes": veto_count, "commits": commit_count, "veto_rate": round(veto_rate, 4)}


def run_experiment():
    print("=" * 70)
    print("Phase 3 v10: Energy-Model Agreement Veto")
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

    # ── Experiment configs ──
    # Compare v9 winner (anneal + antirep p5a1) against veto at different thresholds
    experiments = []

    # Baseline
    experiments.append(("baseline", None, 0, 0, 1, 0.0))

    for topic, target in TOPICS.items():
        # v9 winner: anneal 10→0 + antirep penalty=5 allow=1 (reimplemented here as reference)
        experiments.append((f"{topic}_v9_antirep",  target, 10, 0, 1, -1.0))  # -1 = use antirep instead

        # v10 veto: anneal 10→0, veto thresholds
        experiments.append((f"{topic}_veto01",  target, 10, 0, 1, 0.1))
        experiments.append((f"{topic}_veto03",  target, 10, 0, 1, 0.3))
        experiments.append((f"{topic}_veto05",  target, 10, 0, 1, 0.5))

    results = []

    for name, target, a_start, a_end, gamma, veto_thresh in experiments:
        print(f"[{name}]")

        inputs = tokenizer.apply_chat_template([prompt], add_generation_prompt=True, tokenize=True)
        if isinstance(inputs[0], int):
            inputs = [inputs]

        torch.manual_seed(42)

        if a_start == 0:
            sampler = MDLMSampler(model=model, tokenizer=tokenizer)
            cfg = MDLMSamplerConfig(steps=64, max_new_tokens=64, block_size=32, temperature=0.6)
            outputs = sampler.sample(inputs, cfg, return_dict=True)
            sequences = outputs.sequences
            stats = {"veto_rate": 0.0}
        elif veto_thresh == -1.0:
            # v9 reference: antirep
            sequences = sample_antirep_v9(
                model, tokenizer, inputs,
                steps=64, max_new_tokens=64, block_size=32, temperature=0.6,
                target_texts=target, alpha_start=10, alpha_end=0, gamma=1,
                rep_penalty=5, rep_allowance=1,
            )
            stats = {"veto_rate": -1}
        else:
            sequences, stats = sample_veto(
                model, tokenizer, inputs,
                steps=64, max_new_tokens=64, block_size=32, temperature=0.6,
                target_texts=target,
                alpha_start=a_start, alpha_end=a_end, gamma=gamma,
                veto_threshold=veto_thresh,
            )

        for seq in sequences:
            response = clean_response(tokenizer.decode(seq, skip_special_tokens=False))
            if len(response) < 15:
                continue

            resp_emb = evaluator.encode([response], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
            metrics = {
                "coherence": round(coherence_check(response, evaluator), 4),
                "diversity": round(len(set(response.lower().split())) / max(len(response.split()), 1), 4),
                "veto_rate": stats["veto_rate"],
            }

            if target:
                target_embs = evaluator.encode(target, convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
                metrics["target_sim"] = round(F.cosine_similarity(resp_emb, target_embs).mean().item(), 4)

            s = metrics.get("target_sim", 0)
            metrics["quality"] = round(s * metrics["coherence"] * metrics["diversity"], 4)

            results.append({
                "experiment": name, "target": target is not None,
                "alpha": a_start, "veto_thresh": veto_thresh,
                "response": response, **metrics,
            })

            verdict = "✅" if metrics["quality"] > 0.05 else ("⚠️" if metrics["quality"] > 0.01 else "❌")
            vr = f"veto={stats['veto_rate']:.3f}" if stats["veto_rate"] >= 0 else "antirep"
            print(f"  {verdict} sim={metrics.get('target_sim', '—')} coh={metrics['coherence']:.4f} div={metrics['diversity']:.4f} q={metrics['quality']:.4f} {vr}")
            print(f"    {response[:160]}\n")

    # ── Summary ──
    print(f"\n{'=' * 75}")
    print("v9 ANTIREP vs v10 VETO (by config type)")
    print(f"{'=' * 75}")
    print(f"{'Config':<20} {'sim':>8} {'coh':>8} {'div':>8} {'quality':>8} {'veto':>8}")
    print("─" * 75)

    agg = defaultdict(list)
    for r in results:
        # Group by suffix
        parts = r["experiment"].rsplit("_", 1)
        if len(parts) == 2 and parts[1] in ("antirep", "veto01", "veto03", "veto05"):
            agg[parts[1]].append(r)
        else:
            agg[r["experiment"]].append(r)

    for config in ["antirep", "veto01", "veto03", "veto05"]:
        trials = agg.get(config, [])
        if not trials:
            continue
        sims = [t.get("target_sim", 0) for t in trials]
        cohs = [t["coherence"] for t in trials]
        divs = [t["diversity"] for t in trials]
        quals = [t["quality"] for t in trials]
        vetos = [t["veto_rate"] for t in trials if t["veto_rate"] >= 0]
        vr = f"{np.mean(vetos):.4f}" if vetos else "—"
        print(f"{config:<20} {np.mean(sims):>8.4f} {np.mean(cohs):>8.4f} {np.mean(divs):>8.4f} {np.mean(quals):>8.4f} {vr:>8}")

    # Per-topic
    print(f"\n{'=' * 75}")
    print("BY TOPIC: v9 antirep vs v10 veto")
    print(f"{'=' * 75}")
    for topic in TOPICS:
        print(f"\n  {topic.upper()}:")
        for suffix, label in [
            ("_v9_antirep", "v9 antirep p5a1"),
            ("_veto01",     "v10 veto≥0.1"),
            ("_veto03",     "v10 veto≥0.3"),
            ("_veto05",     "v10 veto≥0.5"),
        ]:
            name = f"{topic}{suffix}"
            trials = [r for r in results if r["experiment"] == name]
            if not trials:
                continue
            t = trials[0]
            vr = f"veto={t['veto_rate']:.3f}" if t["veto_rate"] >= 0 else ""
            print(f"    {label:<22}  sim={t.get('target_sim', 0):.4f}  coh={t['coherence']:.4f}  div={t['diversity']:.4f}  q={t['quality']:.4f}  {vr}")

    # Best outputs
    print(f"\n{'=' * 75}")
    print("TOP 8 OUTPUTS (by quality)")
    print(f"{'=' * 75}")
    for r in sorted(results, key=lambda x: x["quality"], reverse=True)[:8]:
        print(f"\n  [{r['experiment']}] q={r['quality']:.4f}")
        print(f"  sim={r.get('target_sim', 0):.4f} coh={r['coherence']:.4f} div={r['diversity']:.4f} veto={r['veto_rate']}")
        print(f"  {r['response'][:250]}")

    with open(RESULTS_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"{'=' * 75}")


# ── v9 antirep reimplementation for fair comparison ──

def sample_antirep_v9(
    model, tokenizer, inputs, *,
    steps, max_new_tokens, block_size, temperature,
    target_texts, alpha_start, alpha_end, gamma,
    rep_penalty, rep_allowance,
):
    """v9 reference implementation for head-to-head comparison."""
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id
    embed_matrix = model.get_input_embeddings().weight.data.float()

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

    token_counts = torch.zeros(B, embed_matrix.shape[0], device=DEVICE)

    def alpha_at(step):
        progress = step / max(total_steps - 1, 1)
        decay = max(1.0 - progress, 0.0) ** gamma
        return alpha_start * decay + alpha_end * (1.0 - decay)

    def penalty_at(step):
        progress = step / max(total_steps - 1, 1)
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

            with torch.no_grad():
                logits = model(x, attention_mask=attention_mask).logits.float()

            energy_scores = token_scores.unsqueeze(0).unsqueeze(0) * a
            logits = logits + mask_index.unsqueeze(-1).float() * energy_scores

            if pen > 0 and global_step > 2:
                excess = (token_counts - rep_allowance).clamp(min=0)
                penalty_matrix = pen * excess
                logits = logits - penalty_matrix.unsqueeze(1)

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


if __name__ == "__main__":
    run_experiment()
