"""
Phase 3 v13: Fusion — Finding the Optimal Combination of Two Embedding Spaces

PROBLEM:
  Model embeddings (4096D) and MiniLM (384D) rank tokens differently (ρ=0.14).
  Model wins on space/horror, MiniLM wins on ocean/cooking.
  We need ONE scoring function that captures the best of both.

FUSION STRATEGIES TESTED:
  All start from normalized scores in [0,1] or z-score space.

  1. CONVEX:        s = w·m + (1-w)·n          (weighted average, sweep w)
  2. RRF:           s = 1/(k+rank_m) + 1/(k+rank_n)  (reciprocal rank fusion)
  3. GEOMETRIC:     s = sqrt(norm(m) · norm(n))  (requires agreement in both)
  4. HARMONIC:      s = 2·m·n / (m + n + ε)    (penalizes disagreement)
  5. BAYESIAN:      s = softmax(m) · softmax(n) (independent evidence product)
  6. MAX (OR-gate): s = max(m, n)               (union of both signals)

  Each produces a unified token_scores vector. Same sampling (anneal + antirep).
  Same evaluation (target_sim, coherence, diversity, quality).

PRINCIPLE:
  - Geometric/Harmonic/Bayesian → require AGREEMENT (token must be relevant
    in both spaces). Conservative. Should reduce noise.
  - RRF → from Information Retrieval. Proven robust across heterogeneous
    rankers. Parameter k controls top-ranked bias.
  - Convex → simple interpolation. The baseline.
  - Max → permissive. Union of signals. Risk: more noise.

  If one strategy consistently wins across ALL topics, that's our answer.
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
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "results", "phase3_v13_fusion.jsonl")


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


# ── Fusion functions ──

def fuse_convex(m, n, w=0.5):
    """Weighted average. w=0 → pure MiniLM, w=1 → pure Model."""
    return w * m + (1 - w) * n


def fuse_rrf(m, n, k=60):
    """Reciprocal Rank Fusion. Standard IR technique.
    k controls how much top-ranked tokens dominate.
    k=1: only #1 ranked matters. k=60: standard IR. k=500: very flat."""
    rank_m = m.argsort(descending=True).argsort().float()
    rank_n = n.argsort(descending=True).argsort().float()
    return 1.0 / (k + rank_m + 1) + 1.0 / (k + rank_n + 1)


def fuse_geometric(m, n):
    """Geometric mean. Requires token to score well in BOTH spaces."""
    # Normalize to [0, 1] via min-max
    m01 = (m - m.min()) / (m.max() - m.min() + 1e-8)
    n01 = (n - n.min()) / (n.max() - n.min() + 1e-8)
    return torch.sqrt(m01 * n01 + 1e-8)


def fuse_harmonic(m, n):
    """Harmonic mean. Penalizes disagreement more than geometric."""
    m01 = (m - m.min()) / (m.max() - m.min() + 1e-8)
    n01 = (n - n.min()) / (n.max() - n.min() + 1e-8)
    return 2 * m01 * n01 / (m01 + n01 + 1e-8)


def fuse_bayesian(m, n):
    """Bayesian product of independent evidence.
    P(token|topic) ∝ P_model(token|topic) × P_minilm(token|topic)"""
    pm = F.softmax(m * 10, dim=-1)  # scale up for sharper distribution
    pn = F.softmax(n * 10, dim=-1)
    combined = pm * pn
    # Normalize
    combined = combined / (combined.max() + 1e-8)
    return combined


def fuse_max(m, n):
    """OR-gate: take the higher score from either space."""
    return torch.max(m, n)


# ── Compute raw scores from each space ──

def compute_model_scores(model_embed, tokenizer, target_text):
    tokens = tokenizer(target_text, return_tensors="pt", truncation=True, max_length=128)
    ids = tokens["input_ids"].to(DEVICE)
    with torch.no_grad():
        d = F.normalize(model_embed[ids].mean(dim=1).squeeze(0), dim=-1)
        scores = torch.mv(model_embed, d)
        scores = scores / (scores.abs().max() + 1e-8)
    return scores


def compute_minilm_scores(minilm_table, embedder, target_text):
    d = embedder.encode([target_text], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
    d = F.normalize(d.squeeze(0), dim=-1)
    with torch.no_grad():
        scores = torch.mv(minilm_table, d)
        scores = scores / (scores.abs().max() + 1e-8)
    return scores


# ── Sampling (same as v9/v12, parameterized by token_scores) ──

def sample_guided(model, tokenizer, token_scores, inputs, *,
                  steps=64, max_new_tokens=64, block_size=32, temperature=0.6,
                  alpha_start=10, alpha_end=0, gamma=1, rep_penalty=5, rep_allowance=1):
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
        p = step / max(total_steps - 1, 1)
        decay = max(1.0 - p, 0.0) ** gamma
        return alpha_start * decay + alpha_end * (1.0 - decay)

    global_step = 0
    for b in range(num_blocks):
        block_mask = torch.zeros((B, block_size), dtype=torch.bool, device=x.device)
        for j in range(B):
            s = prompt_lens[j] + b * block_size
            e = min(s + block_size, prompt_lens[j] + max_new_tokens, T)
            if s < e:
                block_mask[j, :e - s] = x[j, s:e] == mask_id

        num_transfer = get_num_transfer_tokens(mask_index=block_mask, steps=steps_per_block, scheduler=scheduler, stochastic=False)
        eff_steps = num_transfer.size(1)

        for i in range(eff_steps):
            mask_index = x == mask_id
            a = alpha_at(global_step)
            pen = rep_penalty * (global_step / max(total_steps - 1, 1))

            with torch.no_grad():
                logits = model(x, attention_mask=attention_mask).logits.float()

            logits = logits + mask_index.unsqueeze(-1).float() * (token_scores.unsqueeze(0).unsqueeze(0) * a)

            if pen > 0 and global_step > 2:
                excess = (token_counts - rep_allowance).clamp(min=0)
                logits = logits - (pen * excess).unsqueeze(1)

            x0 = torch.argmax(add_gumbel_noise(logits, temperature=temperature), dim=-1)
            p = F.softmax(logits, dim=-1)
            x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            for j in range(B):
                x0_p[j, prompt_lens[j] + (b + 1) * block_size:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            conf = torch.where(mask_index, x0_p, -np.inf)
            ti = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(B):
                k = int(num_transfer[j, i].item())
                if k > 0:
                    _, sel = torch.topk(conf[j], k=k)
                    ti[j, sel] = True
            for j in range(B):
                cj = x0[j][ti[j]]
                token_counts[j].scatter_add_(0, cj, torch.ones_like(cj, dtype=token_counts.dtype))
            x[ti] = x0[ti]
            global_step += 1

    return x


def run_experiment():
    print("=" * 70)
    print("Phase 3 v13: Fusion — Combining Model + MiniLM Embedding Spaces")
    print("=" * 70)

    model = get_model(model_args=type("Args", (), {
        "model_name_or_path": MODEL_ID, "dtype": torch.bfloat16, "device_map": {"": 0},
    })()).eval()
    tokenizer = get_tokenizer(model_args=type("Args", (), {"model_name_or_path": MODEL_ID})())
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
    model_embed = model.get_input_embeddings().weight.data.float()

    # Build MiniLM token table
    print("\nBuilding MiniLM token table...", flush=True)
    vocab_size = model_embed.shape[0]
    token_texts = [tokenizer.decode([i], skip_special_tokens=True).strip() or "<pad>" for i in range(vocab_size)]
    minilm_table = embedder.encode(token_texts, batch_size=1024, show_progress_bar=False,
                                   convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
    print(f"  Done. MiniLM table: {minilm_table.shape}")

    prompt = [{"role": "user", "content": "Write a short story about something interesting."}]

    TOPICS = {
        "ocean":   "ocean underwater coral reef fish diving deep sea submarine waves",
        "horror":  "horror nightmare monster ghost darkness fear terrifying scream blood",
        "space":   "space exploration stars Mars galaxies astronauts rocket launch mission",
        "cooking": "cooking recipe chef kitchen delicious food spices culinary restaurant",
    }

    # ── Fusion strategies ──
    FUSIONS = [
        ("model_only",    lambda m, n: m),
        ("minilm_only",   lambda m, n: n),
        ("convex_50",     lambda m, n: fuse_convex(m, n, w=0.5)),
        ("convex_30",     lambda m, n: fuse_convex(m, n, w=0.3)),  # favor MiniLM
        ("convex_70",     lambda m, n: fuse_convex(m, n, w=0.7)),  # favor Model
        ("rrf_k10",       lambda m, n: fuse_rrf(m, n, k=10)),
        ("rrf_k60",       lambda m, n: fuse_rrf(m, n, k=60)),
        ("geometric",     lambda m, n: fuse_geometric(m, n)),
        ("harmonic",      lambda m, n: fuse_harmonic(m, n)),
        ("bayesian",      lambda m, n: fuse_bayesian(m, n)),
        ("max",           lambda m, n: fuse_max(m, n)),
    ]

    results = []

    for topic, target_text in TOPICS.items():
        print(f"\n{'─' * 60}")
        print(f"  TOPIC: {topic}")
        print(f"{'─' * 60}")

        m_scores = compute_model_scores(model_embed, tokenizer, target_text)
        n_scores = compute_minilm_scores(minilm_table, embedder, target_text)

        for fusion_name, fusion_fn in FUSIONS:
            # Normalize fused scores
            fused = fusion_fn(m_scores, n_scores)
            fused = fused / (fused.abs().max() + 1e-8)
            fused = fused.to(DEVICE)

            # Show top tokens for first topic
            if topic == "ocean":
                top = fused.topk(8)
                toks = [tokenizer.decode([i]).strip() for i in top.indices]
                print(f"  {fusion_name:<14} top: {toks}")

            # Sample
            torch.manual_seed(42)
            inputs = tokenizer.apply_chat_template([prompt], add_generation_prompt=True, tokenize=True)
            if isinstance(inputs[0], int):
                inputs = [inputs]

            sequences = sample_guided(model, tokenizer, fused, inputs)

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
                    "topic": topic, "fusion": fusion_name,
                    "target_sim": round(sim, 4), "coherence": round(coh, 4),
                    "diversity": round(div, 4), "quality": round(quality, 4),
                    "response": response[:300],
                })

                v = "✅" if quality > 0.08 else ("⚠️" if quality > 0.03 else "❌")
                print(f"    {v} {fusion_name:<14} sim={sim:.4f} coh={coh:.4f} div={div:.4f} q={quality:.4f}")

    # ── Summary ──
    print(f"\n{'=' * 80}")
    print("FUSION STRATEGY COMPARISON")
    print(f"{'=' * 80}")
    print(f"{'Strategy':<14} {'ocean':>10} {'horror':>10} {'space':>10} {'cooking':>10} {'MEAN':>10} {'STD':>8}")
    print("─" * 80)

    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        agg[r["fusion"]][r["topic"]].append(r)

    fusion_means = {}
    for fusion_name, _, in [(f[0], f[1]) for f in FUSIONS]:
        topic_qs = []
        row = f"{fusion_name:<14}"
        for topic in TOPICS:
            trials = agg[fusion_name].get(topic, [])
            if trials:
                q = np.mean([t["quality"] for t in trials])
                topic_qs.append(q)
                row += f" {q:>10.4f}"
            else:
                row += f" {'—':>10}"
        if topic_qs:
            mean_q = np.mean(topic_qs)
            std_q = np.std(topic_qs)
            fusion_means[fusion_name] = (mean_q, std_q)
            row += f" {mean_q:>10.4f} {std_q:>8.4f}"
        print(row)

    # ── Winner ──
    print(f"\n{'=' * 80}")
    if fusion_means:
        sorted_fusions = sorted(fusion_means.items(), key=lambda x: x[1][0], reverse=True)
        print("RANKING (by mean quality across all topics):")
        for i, (name, (mean, std)) in enumerate(sorted_fusions):
            marker = "👑" if i == 0 else f"  {i+1}."
            print(f"  {marker} {name:<14} mean={mean:.4f} std={std:.4f}")

        winner = sorted_fusions[0]
        print(f"\nWINNER: {winner[0]} (mean quality = {winner[1][0]:.4f})")

        # Compare to baselines
        model_q = fusion_means.get("model_only", (0, 0))[0]
        minilm_q = fusion_means.get("minilm_only", (0, 0))[0]
        best_q = winner[1][0]
        print(f"\n  model_only:   {model_q:.4f}")
        print(f"  minilm_only:  {minilm_q:.4f}")
        print(f"  {winner[0]}: {best_q:.4f}")
        print(f"  improvement:  +{best_q - max(model_q, minilm_q):.4f} over best single source")

    # ── Best outputs per fusion ──
    print(f"\n{'=' * 80}")
    print("BEST OUTPUTS PER FUSION STRATEGY")
    print(f"{'=' * 80}")
    best_per = {}
    for r in results:
        f = r["fusion"]
        if f not in best_per or r["quality"] > best_per[f]["quality"]:
            best_per[f] = r

    for fusion_name in [f[0] for f in FUSIONS]:
        if fusion_name in best_per:
            r = best_per[fusion_name]
            print(f"\n  [{fusion_name}] {r['topic']} q={r['quality']:.4f} sim={r['target_sim']:.4f}")
            print(f"  {r['response'][:200]}")

    with open(RESULTS_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    run_experiment()
