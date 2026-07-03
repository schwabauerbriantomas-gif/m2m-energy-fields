"""
Phase 3 v8b: Energy Annealing — Focused Experiment

Reduced from v8: fewer configs, fewer steps, 1 trial each.
Focus: does annealing (10→0) produce better quality than constant (5)?
"""

import sys, time, json, math
import torch, torch.nn.functional as F, numpy as np
from sentence_transformers import SentenceTransformer

import dllm
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.core.samplers.utils import get_num_transfer_tokens, add_gumbel_noise
from dllm.core.schedulers import LinearAlphaScheduler
from dllm.utils import get_model, get_tokenizer

DEVICE = "cuda"
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
RESULTS_FILE = "/root/m2m-energy-fields/results/phase3_v8b_focused.jsonl"


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


def detect_repetition(x, mask_id, prompt_lens, rep_threshold=3):
    B, T = x.shape
    repair_mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
    for j in range(B):
        gen_start = prompt_lens[j]
        gen = x[j, gen_start:]
        for i in range(len(gen)):
            tid = gen[i].item()
            if tid == mask_id:
                continue
            run_len = 1
            for k in range(i + 1, min(i + 20, len(gen))):
                if gen[k].item() == tid:
                    run_len += 1
                else:
                    break
            if run_len >= rep_threshold:
                for k in range(i + 1, i + run_len):
                    if gen[k].item() == tid:
                        repair_mask[j, gen_start + k] = True
    return repair_mask


def sample_guided(
    model, tokenizer, inputs, *,
    steps, max_new_tokens, block_size, temperature,
    target_texts, alpha_start, alpha_end, gamma,
    rep_threshold,
):
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id
    embed_matrix = model.get_input_embeddings().weight.data.float()

    # Energy direction
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
            a = alpha_at(global_step)

            with torch.no_grad():
                logits = model(x, attention_mask=attention_mask).logits

            scores = token_scores.unsqueeze(0).unsqueeze(0) * a
            logits = logits.float() + mask_index.unsqueeze(-1).float() * scores

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

            x[transfer_index] = x0[transfer_index]
            total_commits += transfer_index.sum().item()

            # Repetition repair: re-mask runs of 3+ identical tokens
            if rep_threshold > 0 and global_step > 2:
                rep_mask = detect_repetition(x, mask_id, prompt_lens, rep_threshold)
                num_rep = rep_mask.sum().item()
                if num_rep > 0:
                    x[rep_mask] = mask_id
                    total_repairs += num_rep

            global_step += 1

    return x, {
        "repairs": total_repairs,
        "commits": total_commits,
        "repair_rate": round(total_repairs / max(total_commits, 1), 4),
    }


def run_experiment():
    print("=" * 70)
    print("Phase 3 v8b: Energy Annealing — Focused Test")
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

    # Focused: 4 configs per topic, 1 trial
    experiments = []
    for topic, target in TOPICS.items():
        # C1: constant alpha=5 (best from v4)
        experiments.append(("const5",       topic, target, 5,  5,  1, 0))
        # C2: anneal 10→0 linear (strong start, model finishes)
        experiments.append(("anneal10_0",   topic, target, 10, 0,  1, 0))
        # C3: anneal 10→0 + repair (break repetition loops)
        experiments.append(("anneal10_rep", topic, target, 10, 0,  1, 3))
        # C4: anneal 15→0 gamma=2 + repair (very strong start, slow decay)
        experiments.append(("anneal15_rep", topic, target, 15, 0,  2, 3))

    results = []

    for name, topic, target, a_start, a_end, gamma, rep in experiments:
        label = f"{topic}_{name}"
        print(f"[{label}] a={a_start}→{a_end} γ={gamma} rep={rep}")

        inputs = tokenizer.apply_chat_template([prompt], add_generation_prompt=True, tokenize=True)
        if isinstance(inputs[0], int):
            inputs = [inputs]

        torch.manual_seed(42)
        sequences, stats = sample_guided(
            model, tokenizer, inputs,
            steps=64, max_new_tokens=64, block_size=32, temperature=0.6,
            target_texts=target,
            alpha_start=a_start, alpha_end=a_end, gamma=gamma,
            rep_threshold=rep,
        )

        for seq in sequences:
            response = clean_response(tokenizer.decode(seq, skip_special_tokens=False))
            if len(response) < 15:
                continue
            resp_emb = evaluator.encode([response], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
            target_embs = evaluator.encode(target, convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
            sim = F.cosine_similarity(resp_emb, target_embs).mean().item()
            coh = coherence_check(response, evaluator)
            div = len(set(response.lower().split())) / max(len(response.split()), 1)
            quality = sim * coh * div

            result = {
                "experiment": label, "topic": topic, "config": name,
                "a_start": a_start, "a_end": a_end, "gamma": gamma, "rep": rep,
                "target_sim": round(sim, 4), "coherence": round(coh, 4),
                "diversity": round(div, 4), "quality": round(quality, 4),
                "repair_rate": stats["repair_rate"],
                "response": response,
            }
            results.append(result)
            verdict = "✅" if quality > 0.05 else ("⚠️" if quality > 0.01 else "❌")
            print(f"  {verdict} sim={sim:.4f} coh={coh:.4f} div={div:.4f} q={quality:.4f} repair={stats['repair_rate']:.4f}")
            print(f"    {response[:160]}\n")

    # Summary
    print(f"\n{'=' * 75}")
    print("CONSTANT vs ANNEALED vs ANNEALED+REPAIR")
    print(f"{'=' * 75}")
    print(f"{'Config':<20} {'sim_mean':>9} {'coh_mean':>9} {'div_mean':>9} {'q_mean':>9} {'q_max':>9}")
    print("─" * 75)

    from collections import defaultdict
    by_config = defaultdict(list)
    for r in results:
        by_config[r["config"]].append(r)

    for config in ["const5", "anneal10_0", "anneal10_rep", "anneal15_rep"]:
        trials = by_config.get(config, [])
        if not trials:
            continue
        sims = [t["target_sim"] for t in trials]
        cohs = [t["coherence"] for t in trials]
        divs = [t["diversity"] for t in trials]
        quals = [t["quality"] for t in trials]
        print(f"{config:<20} {np.mean(sims):>9.4f} {np.mean(cohs):>9.4f} {np.mean(divs):>9.4f} {np.mean(quals):>9.4f} {max(quals):>9.4f}")

    # Best outputs
    print(f"\n{'=' * 75}")
    print("TOP 5 OUTPUTS (by quality)")
    print(f"{'=' * 75}")
    for r in sorted(results, key=lambda x: x["quality"], reverse=True)[:5]:
        print(f"\n  [{r['experiment']}] q={r['quality']:.4f}")
        print(f"  sim={r['target_sim']:.4f} coh={r['coherence']:.4f} div={r['diversity']:.4f}")
        print(f"  {r['response'][:250]}")

    with open(RESULTS_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    run_experiment()
