"""Rollout generation (A's first post-switch message), prompted B/C scoring and feature caching for proto_1.

Every rollout is stored as two files with the same `rid` order:
  <name>.public.jsonl   {rid, example_id, message}            <- the only thing B/C-side code reads
  <name>.private.jsonl  {rid, example_id, method, action, rollout, a_system, a_user, raw}
"""
import numpy as np

from . import agents
from .config import read_jsonl, stable_hash, sub, write_jsonl
from .data import load_split


def gen_settings(cfg):
    g = cfg["gen"]
    return dict(max_new_tokens=g["max_new_tokens"], temperature=g["temperature"], top_p=g["top_p"],
                do_sample=True, batch_size=g["batch_size"])


def generate(cfg, spec, lm, split, jobs, seed_tag):
    """jobs: list of dicts {example_id, method, action(or None), rollout}."""
    pub, lab = load_split(cfg, split)
    hist = {r["example_id"]: r["history"] for r in pub}
    prompts, prefixes, private = [], [], []
    for j in jobs:
        eid = j["example_id"]
        system, user, prefix = agents.a_prompt(spec, j["method"], hist[eid], lab[eid], j.get("action"), eid)
        prompts.append(lm.chat(system, user))
        prefixes.append(prefix)
        private.append({"rid": stable_hash(seed_tag, eid, j["method"], j.get("action"), j["rollout"]),
                        "example_id": eid, "method": j["method"], "action": j.get("action"),
                        "rollout": j["rollout"], "a_system": system, "a_user": user})
    seed = int(stable_hash(cfg["seed"], seed_tag, n=6), 16)
    raw = lm.generate(prompts, seed=seed, desc=f"gen {seed_tag}", **gen_settings(cfg))
    public = []
    for p, pre, r in zip(private, prefixes, raw):
        p["raw"] = r
        public.append({"rid": p["rid"], "example_id": p["example_id"], "message": agents.clean_message(pre, r)})
    assert len({p["rid"] for p in public}) == len(public), "rollout id collision"
    return public, private


def save(cfg, split, name, public, private):
    write_jsonl(sub(cfg, "rollouts", split, f"{name}.public.jsonl"), public)
    write_jsonl(sub(cfg, "rollouts", split, f"{name}.private.jsonl"), private)


def load_public(cfg, split, name):
    return read_jsonl(sub(cfg, "rollouts", split, f"{name}.public.jsonl"))


def load_private(cfg, split, name):
    return read_jsonl(sub(cfg, "rollouts", split, f"{name}.private.jsonl"))


def classify(cfg, lm, prompts, desc):
    c = cfg["classify"]
    raw = lm.generate([lm.chat(s, u) for s, u in prompts], max_new_tokens=c["max_new_tokens"],
                      do_sample=False, batch_size=c["batch_size"], seed=0, desc=desc)
    return raw


def score_c(cfg, spec, lm, public):
    """C's fixed prompted classifier: sees only c_view(message)."""
    raw = classify(cfg, lm, [agents.c_prompt(spec, r["message"]) for r in public], "C")
    K = len(spec["hidden_topics"])
    return [{"rid": r["rid"], "pred": agents.parse_label(x, K), "raw": x} for r, x in zip(public, raw)]


def score_b_prompted(cfg, spec, lm, public, history_by_id, mode):
    prompts = [agents.b_prompt(spec, history_by_id[r["example_id"]], r.get("message"), mode) for r in public]
    raw = classify(cfg, lm, prompts, f"B-{mode}")
    K = len(spec["hidden_topics"])
    return [{"rid": r.get("rid", r["example_id"]), "pred": agents.parse_label(x, K), "raw": x}
            for r, x in zip(public, raw)]


def message_features(cfg, lm, public):
    f = cfg["features"]
    return lm.features([agents.c_view(r["message"]) for r in public], f["layers"], f["batch_size"])


def history_features(cfg, lm, split):
    """Frozen features of B's permitted history text, cached per split (aligned to public order)."""
    path = sub(cfg, "features", f"{split}_hist.npy")
    pub, _ = load_split(cfg, split)
    if path.exists():
        return np.load(path), [r["example_id"] for r in pub]
    f = cfg["features"]
    feats = lm.features([agents.render_history(r["history"]) for r in pub], f["layers"], f["batch_size"])
    np.save(path, feats)
    return feats, [r["example_id"] for r in pub]
