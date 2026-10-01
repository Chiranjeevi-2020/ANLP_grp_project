"""Synthetic dataset construction with family-level train/dev/test splits.

Public records (what B may see as history) and private labels (gold z, fills,
family) are written to SEPARATE files. Example ids are opaque hashes.
"""
import itertools
import random
import time
import uuid

from .config import SPLITS, load_spec, read_jsonl, stable_hash, sub, write_json, write_jsonl


def fill_template(turns, fill):
    return [t.format(**fill) for t in turns]


def build_examples(cfg, spec):
    rng = random.Random(cfg["seed"])
    split_of = {i: s for s, idxs in spec["split_by_template_index"].items() for i in idxs}
    topics_by_pair = {}
    for t in spec["hidden_topics"]:
        topics_by_pair.setdefault(t["pair"], []).append(t["id"])
    fills = spec["fills"]
    combos = list(itertools.product(fills["person"], fills["day"], fills["club"]))
    public = {s: [] for s in SPLITS}
    private = {s: [] for s in SPLITS}
    for pair, templates in spec["history_templates"].items():
        pair = int(pair)
        for ti, turns in enumerate(templates):
            split = split_of[ti]
            family = f"pair{pair}_tpl{ti}"
            n_fill = cfg["data"]["fills_per_template"][split]
            for fi, (person, day, club) in enumerate(rng.sample(combos, n_fill)):
                fill = {"person": person, "day": day, "club": club}
                text = fill_template(turns, fill)
                history = [["A" if k % 2 == 0 else "B", u] for k, u in enumerate(text)]
                # SAME history for both topics of the pair: history alone cannot separate them.
                for z in topics_by_pair[pair]:
                    eid = stable_hash(cfg["seed"], family, fi, z, rng.random())
                    public[split].append({"example_id": eid, "history": history})
                    private[split].append({"example_id": eid, "z": z, "pair": pair,
                                           "family": family, "fill": fill})
    for s in SPLITS:  # shuffle so file order does not reveal labels
        order = list(range(len(public[s])))
        rng.shuffle(order)
        public[s] = [public[s][i] for i in order]
        private[s] = [private[s][i] for i in order]
    return public, private


def char_ngrams(text, n=5):
    text = " ".join(text.lower().split())
    return {text[i:i + n] for i in range(max(1, len(text) - n + 1))}


def history_text(history):
    return "\n".join(f"{s}: {u}" for s, u in history)


def duplicate_report(public):
    """Exact and near-duplicate (char-5gram Jaccard) check across splits."""
    texts = {s: {history_text(r["history"]) for r in public[s]} for s in SPLITS}
    exact, max_j, worst = 0, 0.0, None
    grams = {s: [(t, char_ngrams(t)) for t in texts[s]] for s in SPLITS}
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        exact += len(texts[a] & texts[b])
        for ta, ga in grams[a]:
            for tb, gb in grams[b]:
                j = len(ga & gb) / max(1, len(ga | gb))
                if j > max_j:
                    max_j, worst = j, (a, b, ta[:80], tb[:80])
    return {"exact_cross_split_duplicates": exact, "max_cross_split_jaccard": round(max_j, 4),
            "most_similar_pair": worst}


def build(cfg):
    spec = load_spec()
    public, private = build_examples(cfg, spec)
    for s in SPLITS:
        write_jsonl(sub(cfg, "data", f"{s}_public.jsonl"), public[s])
        write_jsonl(sub(cfg, "data", f"{s}_labels.jsonl"), private[s])
    counts = {s: {} for s in SPLITS}
    for s in SPLITS:
        for r in private[s]:
            counts[s][r["z"]] = counts[s].get(r["z"], 0) + 1
    report = {"n_examples": {s: len(public[s]) for s in SPLITS},
              "per_topic_counts": counts,
              "families": {s: sorted({r["family"] for r in private[s]}) for s in SPLITS},
              **duplicate_report(public)}
    write_json(sub(cfg, "manifest.json"),
               {"run_id": uuid.uuid4().hex[:12], "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "config": {k: v for k, v in cfg.items() if not k.startswith("_")}})
    write_json(sub(cfg, "data", "data_report.json"), report)
    print(f"[data] {report['n_examples']}  exact_dups={report['exact_cross_split_duplicates']} "
          f"max_jaccard={report['max_cross_split_jaccard']}")
    return report


def load_split(cfg, split):
    """Returns (public_rows, labels_by_example_id). Callers must keep these separate."""
    pub = read_jsonl(sub(cfg, "data", f"{split}_public.jsonl"))
    lab = {r["example_id"]: r for r in read_jsonl(sub(cfg, "data", f"{split}_labels.jsonl"))}
    assert [r["example_id"] for r in pub] and all(r["example_id"] in lab for r in pub)
    return pub, lab
