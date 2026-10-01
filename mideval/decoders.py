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
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def select_and_train(kind, tr, dev, n_labels, layers, cfg):
    """tr/dev: dicts with hist, msg, y, sel_mask (dev rows used for selection). Returns model, info."""
    d = cfg["decoder"]
    grid = []
    for li, layer in enumerate(layers):
        Xtr, Xdv = make_inputs(kind, tr["hist"], tr["msg"], li), make_inputs(kind, dev["hist"], dev["msg"], li)
        for wd in d["weight_decays"]:
            m = fit(Xtr, tr["y"], n_labels, wd, d["epochs"], d["lr"], cfg["seed"])
            pred = m.predict(Xdv)
            acc_sel = float((pred[dev["sel"]] == dev["y"][dev["sel"]]).mean())
            grid.append({"layer": layer, "layer_idx": li, "weight_decay": wd, "dev_sel_acc": acc_sel,
                         "train_acc": float((m.predict(Xtr) == tr["y"]).mean())})
    best = max(grid, key=lambda g: (g["dev_sel_acc"], -g["weight_decay"]))
    li = best["layer_idx"]
    model = fit(make_inputs(kind, tr["hist"], tr["msg"], li), tr["y"], n_labels,
                best["weight_decay"], d["epochs"], d["lr"], cfg["seed"])
    return model, {"best": best, "grid": grid}


def save(path, model, info):
    torch.save({"state_dict": model.state_dict(), "dim": model.linear.in_features,
                "n_labels": model.linear.out_features, "info": info, "checksum": model.checksum()}, path)


def load(path):
    ck = torch.load(path, weights_only=False)
    sd = ck["state_dict"]
    m = LinearDecoder(ck["dim"], ck["n_labels"], sd["mean"], sd["std"])
    m.load_state_dict(sd)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    assert m.checksum() == ck["checksum"], "decoder checkpoint corrupted"
    return m, ck["info"]
