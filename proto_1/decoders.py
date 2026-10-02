"""Supervised B decoder (and matched-architecture control decoders) on frozen LM features.

kind = "full"     : [history feature ; message feature]   (B's permitted input; used as RL reward)
kind = "history"  : history feature only                  (control 1: B history-only)
kind = "message"  : message feature only                  (control 2 and 4: B message-only ==
                     public-transcript decoder, because C's public view is exactly the message view)
Trained on TRAIN-split rollouts of the baseline methods only; layer and weight decay chosen on DEV.
"""
import hashlib

import numpy as np
import torch
import torch.nn as nn

KINDS = ("full", "history", "message")


class LinearDecoder(nn.Module):
    def __init__(self, dim, n_labels, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.linear = nn.Linear(dim, n_labels)

    def forward(self, x):
        return self.linear((x - self.mean) / self.std)

    @torch.no_grad()
    def predict(self, x):
        return self.forward(torch.as_tensor(x, dtype=torch.float32)).argmax(-1).numpy()

    def checksum(self):
        h = hashlib.sha1()
        for v in self.state_dict().values():
            h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()[:16]


def make_inputs(kind, hist, msg, layer_idx):
    """hist, msg: [n, n_layers, H] arrays aligned row-by-row (hist already gathered per rollout)."""
    if kind == "full":
        return np.concatenate([hist[:, layer_idx], msg[:, layer_idx]], -1).astype(np.float32)
    if kind == "history":
        return hist[:, layer_idx].astype(np.float32)
    if kind == "message":
        return msg[:, layer_idx].astype(np.float32)
    raise ValueError(kind)


def fit(Xtr, ytr, n_labels, weight_decay, epochs, lr, seed):
    torch.manual_seed(seed)
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-4
    model = LinearDecoder(Xtr.shape[1], n_labels, mean, std)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    X, y = torch.as_tensor(Xtr), torch.as_tensor(ytr, dtype=torch.long)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(X), y)
        loss.backward()
        opt.step()
    return model


def select_and_train(kind, tr, dev, n_labels, layers, cfg):
    """Grid search over layer x weight_decay; evaluates ONLY on dev['sel'] (the switched rollouts)."""
    d = cfg["decoder"]
    best_acc, best_key, best_model = -1.0, None, None
    grid = []
    for li, layer in enumerate(layers):
        Xtr = make_inputs(kind, tr["hist"], tr["msg"], li)
        Xdev = make_inputs(kind, dev["hist"], dev["msg"], li)[dev["sel"]]
        ydev = dev["y"][dev["sel"]]
        for wd in d["weight_decays"]:
            m = fit(Xtr, tr["y"], n_labels, wd, d["epochs"], d["lr"], cfg["seed"])
            acc = float((m.predict(Xdev) == ydev).mean())
            grid.append({"layer": int(layer), "layer_idx": li, "weight_decay": float(wd), "dev_acc_switched": acc})
            if acc > best_acc:
                best_acc, best_key, best_model = acc, grid[-1], m
    return best_model, {"kind": kind, "best": best_key, "grid": grid}


def save(path, model, info):
    torch.save({"state_dict": model.state_dict(), "info": info}, path)


def load(path):
    d = torch.load(path, weights_only=False)
    sd = d["state_dict"]
    dim = sd["linear.weight"].shape[1]
    n_labels = sd["linear.weight"].shape[0]
    m = LinearDecoder(dim, n_labels, sd["mean"], sd["std"])
    m.load_state_dict(sd)
    return m.eval(), d["info"]
