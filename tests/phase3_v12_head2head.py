"""
Phase 3 v12: TRUE Head-to-Head — Model Embeddings vs MiniLM Semantic

v11 had a bug: both 'v9_token' and 'v11_semantic' used the same
SemanticEnergyField class → identical results.

This script runs two GENUINELY DIFFERENT energy sources:
  - MODEL:  scores[v] = dot(model_embed[v], d_model)     where d_model is from model embeddings 4096D
  - MINILM: scores[v] = dot(minilm_embed[v], d_minilm)  where d_minilm is from MiniLM 384D

Both use identical sampling: anneal 10→0 + anti-rep p5a1
The ONLY difference is which embedding space computes the energy.
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
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "results", "phase3_v12_head2head.jsonl")


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


def build_token_table_minilm(embedder, tokenizer, vocab_size):
    """MiniLM embedding for every token: [vocab, 384]"""
    print("  Building MiniLM token table...", flush=True)
    t0 = time.time()
    token_texts = []
    for tid in range(vocab_size):
        try:
            text = tokenizer.decode([tid], skip_special_tokens=True).strip()
        except Exception:
            text = ""
        token_texts.append(text if text else "<pad>")
    embs = embedder.encode(
        token_texts, batch_size=1024, show_progress_bar=False,
        convert_to_tensor=True, normalize_embeddings=True, device=DEVICE,
    )
    print(f"  Done in {time.time()-t0:.1f}s. Shape: {embs.shape}", flush=True)
    return embs


def build_token_table_model(model_embed, target_tokens_ids):
    """Model embedding for every token: [vocab, hidden_dim]"""
    return model_embed  # already [vocab, 4096], no preprocessing needed


def compute_direction_model(model_embed, tokenizer, target_text):
    """Energy direction using MODEL embeddings (4096D)."""
    tokens = tokenizer(target_text, return_tensors="pt", truncation=True, max_length=128)
    ids = tokens["input_ids"].to(DEVICE)
    with torch.no_grad():
        d = model_embed[ids].mean(dim=1).squeeze(0)
        d = F.normalize(d, dim=-1)
        scores = torch.mv(model_embed, d)
        scores = scores / (scores.abs().max() + 1e-8)
    return scores


def compute_direction_minilm(minilm_table, embedder, target_text):
    """Energy direction using MINILM embeddings (384D)."""
    d = embedder.encode([target_text], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
    d = F.normalize(d.squeeze(0), dim=-1)
    with torch.no_grad():
        scores = torch.mv(minilm_table, d)
        scores = scores / (scores.abs().max() + 1e-8)
    return scores


def sample_head2head(
    model, tokenizer, token_scores, inputs, *,
    steps, max_new_tokens, block_size, temperature,
    alpha_start, alpha_end, gamma,
    rep_penalty, rep_allowance,
):
    """MDLM sampling with pre-computed token_scores + annealing + anti-rep."""
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id

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

    token_counts = torch.zeros(B, token_scores.shape[0], device=DEVICE)

    def alpha_at(step):
        progress = step / max(total_steps - 1, 1)
        return alpha_start * max(1.0 - progress, 0.0) ** gamma + alpha_end * (1.0 - max(1.0 - progress, 0.0) ** gamma)

    def penalty_at(step):
        return rep_penalty * (step / max(total_steps - 1, 1))

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

            # Energy guidance
            scores = token_scores.unsqueeze(0).unsqueeze(0) * a
            logits = logits + mask_index.unsqueeze(-1).float() * scores

            # Anti-rep
            if pen > 0 and global_step > 2:
                excess = (token_counts - rep_allowance).clamp(min=0)
                logits = logits - (pen * excess).unsqueeze(1)

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
                committed_j = x0[j][transfer_index[j]]
                token_counts[j].scatter_add_(
                    0, committed_j,
                    torch.ones_like(committed_j, dtype=token_counts.dtype),
                )

            x[transfer_index] = x0[transfer_index]
            global_step += 1

    return x


def run_experiment():
    print("=" * 70)
    print("Phase 3 v12: TRUE Head-to-Head")
    print("Model Embeddings (4096D) vs MiniLM Semantic (384D)")
    print("=" * 70)

    print("\n[1] Loading LLaDA-8B...", flush=True)
    model = get_model(model_args=type("Args", (), {
        "model_name_or_path": MODEL_ID, "dtype": torch.bfloat16, "device_map": {"": 0},
    })()).eval()
    tokenizer = get_tokenizer(model_args=type("Args", (), {"model_name_or_path": MODEL_ID})())

    print("\n[2] Loading MiniLM...", flush=True)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)

    print("\n[3] Building token tables...", flush=True)
    model_embed = model.get_input_embeddings().weight.data.float()  # [126464, 4096]
    vocab_size = model_embed.shape[0]
    minilm_table = build_token_table_minilm(embedder, tokenizer, vocab_size)

    print(f"\n  Model embed:  {model_embed.shape}")
    print(f"  MiniLM table: {minilm_table.shape}")

    prompt = [{"role": "user", "content": "Write a short story about something interesting."}]

    TOPICS = {
        "ocean":   "ocean underwater coral reef fish diving deep sea submarine waves",
        "horror":  "horror nightmare monster ghost darkness fear terrifying scream blood",
        "space":   "space exploration stars Mars galaxies astronauts rocket launch mission",
        "cooking": "cooking recipe chef kitchen delicious food spices culinary restaurant",
    }

    results = []
    N_TRIALS = 2

    # Baseline first
    print(f"\n[baseline]")
    inputs = tokenizer.apply_chat_template([prompt], add_generation_prompt=True, tokenize=True)
    if isinstance(inputs[0], int):
        inputs = [inputs]
    sampler = MDLMSampler(model=model, tokenizer=tokenizer)
    cfg = MDLMSamplerConfig(steps=64, max_new_tokens=64, block_size=32, temperature=0.6)
    torch.manual_seed(42)
    outputs = sampler.sample(inputs, cfg, return_dict=True)
    for seq in outputs.sequences:
        response = clean_response(tokenizer.decode(seq, skip_special_tokens=False))
        resp_emb = embedder.encode([response], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
        coh = coherence_check(response, embedder)
        div = len(set(response.lower().split())) / max(len(response.split()), 1)
        results.append({"experiment": "baseline", "response": response,
                        "coherence": round(coh, 4), "diversity": round(div, 4),
                        "quality": 0.0})
        print(f"  coh={coh:.4f} div={div:.4f}")
        print(f"    {response[:160]}")

    # Head-to-head per topic
    for topic, target_text in TOPICS.items():
        # Pre-compute both directions
        scores_model = compute_direction_model(model_embed, tokenizer, target_text)
        scores_minilm = compute_direction_minilm(minilm_table, embedder, target_text)

        # Show difference in top tokens
        top_model = scores_model.topk(8)
        top_minilm = scores_minilm.topk(8)
        model_toks = [tokenizer.decode([i]).strip() for i in top_model.indices]
        minilm_toks = [tokenizer.decode([i]).strip() for i in top_minilm.indices]

        print(f"\n{'─' * 60}")
        print(f"  TOPIC: {topic}")
        print(f"{'─' * 60}")
        print(f"  Model top:  {model_toks}")
        print(f"  MiniLM top: {minilm_toks}")

        # Show tokens that MiniLM catches but Model doesn't
        model_top50 = set(scores_model.topk(50).indices.tolist())
        minilm_top50 = set(scores_minilm.topk(50).indices.tolist())
        only_minilm = minilm_top50 - model_top50
        only_minilm_toks = [tokenizer.decode([i]).strip() for i in list(only_minilm)[:8]]
        print(f"  MiniLM-only: {only_minilm_toks}")

        for source_name, token_scores in [("model", scores_model), ("minilm", scores_minilm)]:
            label = f"{topic}_{source_name}"

            for trial in range(N_TRIALS):
                torch.manual_seed(42 + trial)
                inputs = tokenizer.apply_chat_template([prompt], add_generation_prompt=True, tokenize=True)
                if isinstance(inputs[0], int):
                    inputs = [inputs]

                sequences = sample_head2head(
                    model, tokenizer, token_scores, inputs,
                    steps=64, max_new_tokens=64, block_size=32, temperature=0.6,
                    alpha_start=10, alpha_end=0, gamma=1,
                    rep_penalty=5, rep_allowance=1,
                )

                for seq in sequences:
                    response = clean_response(tokenizer.decode(seq, skip_special_tokens=False))
                    if len(response) < 15:
                        continue

                    resp_emb = embedder.encode([response], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
                    target_embs = embedder.encode([target_text], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
                    sim = F.cosine_similarity(resp_emb, target_embs).mean().item()
                    coh = coherence_check(response, embedder)
                    div = len(set(response.lower().split())) / max(len(response.split()), 1)
                    quality = sim * coh * div

                    results.append({
                        "experiment": label, "topic": topic, "source": source_name,
                        "trial": trial, "response": response,
                        "target_sim": round(sim, 4), "coherence": round(coh, 4),
                        "diversity": round(div, 4), "quality": round(quality, 4),
                    })

                    verdict = "✅" if quality > 0.05 else ("⚠️" if quality > 0.01 else "❌")
                    print(f"  [{label} t{trial}] {verdict} sim={sim:.4f} coh={coh:.4f} div={div:.4f} q={quality:.4f}")
                    print(f"    {response[:180]}")

    # ── Summary ──
    print(f"\n{'=' * 75}")
    print("HEAD-TO-HEAD: MODEL EMBEDDINGS vs MINILM SEMANTIC")
    print(f"{'=' * 75}")
    print(f"{'Topic':<12} {'Source':<8} {'sim_mean':>9} {'sim_max':>9} {'coh_mean':>9} {'div_mean':>9} {'q_mean':>9}")
    print("─" * 75)

    agg = defaultdict(list)
    for r in results:
        if "target_sim" in r:
            agg[(r["topic"], r["source"])].append(r)

    for topic in TOPICS:
        for source in ["model", "minilm"]:
            trials = agg.get((topic, source), [])
            if not trials:
                continue
            sims = [t["target_sim"] for t in trials]
            cohs = [t["coherence"] for t in trials]
            divs = [t["diversity"] for t in trials]
            quals = [t["quality"] for t in trials]
            print(f"{topic:<12} {source:<8} {np.mean(sims):>9.4f} {max(sims):>9.4f} {np.mean(cohs):>9.4f} {np.mean(divs):>9.4f} {np.mean(quals):>9.4f}")
        # Delta
        m = agg.get((topic, "model"), [])
        n = agg.get((topic, "minilm"), [])
        if m and n:
            delta_sim = np.mean([t["target_sim"] for t in n]) - np.mean([t["target_sim"] for t in m])
            delta_q = np.mean([t["quality"] for t in n]) - np.mean([t["quality"] for t in m])
            winner = "MiniLM" if delta_q > 0 else "Model"
            print(f"{'':<12} {'Δ':<8} {delta_sim:>+9.4f} {'':>9} {'':>9} {'':>9} {delta_q:>+9.4f}  → {winner}")
        print()

    # Overall
    all_model = [r for r in results if r.get("source") == "model"]
    all_minilm = [r for r in results if r.get("source") == "minilm"]
    if all_model and all_minilm:
        ms = np.mean([r["target_sim"] for r in all_model])
        ns = np.mean([r["target_sim"] for r in all_minilm])
        mq = np.mean([r["quality"] for r in all_model])
        nq = np.mean([r["quality"] for r in all_minilm])
        print(f"{'OVERALL':<12} {'model':<8} {ms:>9.4f} {'':>9} {'':>9} {'':>9} {mq:>9.4f}")
        print(f"{'OVERALL':<12} {'minilm':<8} {ns:>9.4f} {'':>9} {'':>9} {'':>9} {nq:>9.4f}")
        print(f"{'OVERALL':<12} {'Δ':<8} {ns-ms:>+9.4f} {'':>9} {'':>9} {'':>9} {nq-mq:>+9.4f}")

    # Best outputs per source
    print(f"\n{'=' * 75}")
    print("BEST MINILM OUTPUTS")
    print(f"{'=' * 75}")
    minilm_sorted = sorted(all_minilm, key=lambda x: x["quality"], reverse=True)
    for r in minilm_sorted[:4]:
        print(f"\n  [{r['experiment']}] q={r['quality']:.4f} sim={r['target_sim']:.4f}")
        print(f"  {r['response'][:300]}")

    print(f"\n{'=' * 75}")
    print("BEST MODEL OUTPUTS")
    print(f"{'=' * 75}")
    model_sorted = sorted(all_model, key=lambda x: x["quality"], reverse=True)
    for r in model_sorted[:4]:
        print(f"\n  [{r['experiment']}] q={r['quality']:.4f} sim={r['target_sim']:.4f}")
        print(f"  {r['response'][:300]}")

    with open(RESULTS_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    run_experiment()
