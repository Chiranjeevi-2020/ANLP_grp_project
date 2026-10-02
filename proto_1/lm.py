"""Frozen local LM used for A (generation), prompted B/C (classification), features, and perplexity scoring.

Includes robust device auto-detection (cuda > mps > cpu) and automatic dtype selection.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(requested_device="auto"):
    """Robust device resolution prioritizing: cuda > mps > cpu."""
    if requested_device in ("auto", None) or requested_device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested_device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return requested_device


def resolve_dtype(device, requested_dtype=None):
    """Automatically selects float16 for CUDA/MPS and float32 for CPU."""
    if requested_dtype:
        if device == "cpu" and requested_dtype in ("float16", "bfloat16"):
            return "float32"
        return requested_dtype
    if device in ("cuda", "mps"):
        return "float16"
    return "float32"


class FrozenLM:
    def __init__(self, name, device="auto", dtype=None):
        self.device = resolve_device(device)
        self.dtype_str = resolve_dtype(self.device, dtype)
        torch_dtype = getattr(torch, self.dtype_str)

        print(f"[FrozenLM] Initializing '{name}' on [{self.device}] with dtype [{self.dtype_str}]...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch_dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(name, dtype=torch_dtype)

        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden = self.model.config.hidden_size

    def chat(self, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _batches(self, texts, bs):
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        for b in range(0, len(order), bs):
            yield b // bs, order[b:b + bs]

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens, temperature=0.8, top_p=0.95, do_sample=True,
                 batch_size=32, seed=0, desc=""):
        out = [None] * len(prompts)
        n_batches = (len(prompts) + batch_size - 1) // batch_size
        for bi, idx in self._batches(prompts, batch_size):
            torch.manual_seed(seed * 100003 + bi)
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
            kw = dict(do_sample=True, temperature=temperature, top_p=top_p) if do_sample else \
                dict(do_sample=False, temperature=None, top_p=None, top_k=None)
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tok.pad_token_id, **kw)
            texts = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, t in zip(idx, texts):
                out[i] = t
            if desc and (bi % 10 == 0 or bi == n_batches - 1):
                print(f"  [{desc}] batch {bi + 1}/{n_batches}", flush=True)
        return out

    @torch.no_grad()
    def features(self, texts, layers, batch_size=32):
        """Mean-pooled hidden states of plain text at the given layers -> [n, len(layers), H] float16."""
        out = np.zeros((len(texts), len(layers), self.hidden), dtype=np.float16)
        for _, idx in self._batches(texts, batch_size):
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
            hs = self.model(**enc, output_hidden_states=True).hidden_states
            m = enc["attention_mask"].unsqueeze(-1).float()
            for j, layer in enumerate(layers):
                pooled = (hs[layer].float() * m).sum(1) / m.sum(1)
                out[idx, j] = pooled.cpu().numpy().astype(np.float16)
        return out

    @torch.no_grad()
    def compute_loss_and_perplexity(self, context_prompts, target_texts, batch_size=16):
        """Calculates token cross-entropy loss and perplexity of target_texts given context_prompts."""
        losses = []
        for i in range(0, len(context_prompts), batch_size):
            c_batch = context_prompts[i:i + batch_size]
            t_batch = target_texts[i:i + batch_size]
            full_texts = [c + t for c, t in zip(c_batch, t_batch)]

            enc_c = self.tok(c_batch, add_special_tokens=False)
            enc_full = self.tok(full_texts, return_tensors="pt", padding=True).to(self.device)

            labels = enc_full["input_ids"].clone()
            for b_idx, c_toks in enumerate(enc_c["input_ids"]):
                c_len = len(c_toks)
                pad_len = int((enc_full["attention_mask"][b_idx] == 0).sum().item())
                labels[b_idx, :pad_len + c_len] = -100
                labels[b_idx, enc_full["attention_mask"][b_idx] == 0] = -100

            outputs = self.model(**enc_full, labels=labels)
            losses.append(outputs.loss.item())

        mean_loss = float(np.mean(losses)) if losses else 0.0
        ppl = float(np.exp(min(mean_loss, 20.0)))
        return {"loss": mean_loss, "perplexity": ppl}

    @torch.no_grad()
    def checksum(self):
        s = a = 0.0
        for p in self.model.parameters():
            s += p.float().sum().item()
            a += p.float().abs().sum().item()
        return f"{s:.6e}|{a:.6e}"

    def any_requires_grad(self):
        return any(p.requires_grad for p in self.model.parameters())
