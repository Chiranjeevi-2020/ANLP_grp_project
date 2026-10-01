"""Frozen local LM used for A (generation), prompted B/C (classification) and features."""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class FrozenLM:
    def __init__(self, name, device="mps", dtype="float16"):
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        if device == "cpu":
            dtype = "float32"
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype))
        except TypeError:  # older transformers
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch, dtype))
        self.model = model.to(device).eval()
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
    def checksum(self):
        s = a = 0.0
        for p in self.model.parameters():
            s += p.float().sum().item()
            a += p.float().abs().sum().item()
        return f"{s:.6e}|{a:.6e}"

    def any_requires_grad(self):
        return any(p.requires_grad for p in self.model.parameters())
