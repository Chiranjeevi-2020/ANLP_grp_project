"""Final evaluation on the untouched TEST split: baselines vs controllers + controls for proto_1.

Includes WandB metric tracking and optional Hugging Face Hub uploads.
"""
import csv
import math

import numpy as np
import torch

from . import decoders, rollouts
from .config import read_json, stamp, sub, write_json, write_jsonl
from .data import load_split
from .pipeline import _policy_inputs, load_policy, rewards_from
from .tracker import ExperimentTracker, upload_to_huggingface

METHODS = ["no_switch", "scripted", "prompt_only", "untrained_controller", "rl_controller", "rl_controller_greedy"]


def wilson(k, n, zq=1.96):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    d = 1 + zq ** 2 / n
    c = (p + zq ** 2 / (2 * n)) / d
    h = zq * math.sqrt(p * (1 - p) / n + zq ** 2 / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def cluster_bootstrap(values, clusters, n_boot, seed):
    """95% CI of the mean, resampling test EXAMPLES (rollouts of one example are not independent)."""
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    uniq = np.unique(clusters)
    groups = [values[clusters == u] for u in uniq]
    rng = np.random.default_rng(seed)
    means = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
             for _ in range(n_boot)]
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def acc_block(correct):
    correct = np.asarray(correct, bool)
    return {"acc": round(float(correct.mean()), 4), "k": int(correct.sum()), "n": int(len(correct)),
            "ci95": wilson(int(correct.sum()), len(correct))}


def _jobs(cfg, spec, pub, lab, method, policies, feats):
    n = cfg["eval"]["rollouts_per_example"]
    if method in ("no_switch", "scripted", "prompt_only"):
        return [{"example_id": r["example_id"], "method": method, "rollout": k} for r in pub for k in range(n)]
    policy = policies["init" if method == "untrained_controller" else "best"]
    z = torch.as_tensor([lab[r["example_id"]]["z"] for r in pub])
    with torch.no_grad():
        probs = policy.dist(feats, z).probs
    gen = torch.Generator().manual_seed(cfg["seed"] + 7)
    jobs = []
    for i, r in enumerate(pub):
        for k in range(n):
            a = int(probs[i].argmax()) if method.endswith("greedy") else int(torch.multinomial(probs[i], 1, generator=gen))
            jobs.append({"example_id": r["example_id"], "method": "controller", "action": a, "rollout": k})
    return jobs


def evaluate(cfg, spec, lm, tracker=None, methods=None):
    tracker_created = False
    if tracker is None:
        tracker = ExperimentTracker(cfg, job_type="eval")
        tracker_created = True

    if methods is None:
        methods = METHODS

    K = len(spec["hidden_topics"])
    pub, lab = load_split(cfg, "test")
    hist_by_id = {r["example_id"]: r["history"] for r in pub}
    hist, ids = rollouts.history_features(cfg, lm, "test")
    pos = {e: i for i, e in enumerate(ids)}
    dec = {k: decoders.load(sub(cfg, "decoders", f"{k}.pt")) for k in decoders.KINDS}
    li = {k: dec[k][1]["best"]["layer_idx"] for k in decoders.KINDS}
    ftr = _policy_inputs(cfg, "train", li["full"])
    policies = {"init": load_policy(cfg, spec, sub(cfg, "rl", "controller_init.pt"), ftr),
                "best": load_policy(cfg, spec, sub(cfg, "rl", "controller_best.pt"), ftr)}
    feats_test = _policy_inputs(cfg, "test", li["full"])
    lam = cfg["rl"]["lambda"]
    rng = np.random.default_rng(cfg["seed"] + 11)

    # method-independent controls (per test example)
    hist_dec_pred = dec["history"][0].predict(hist[:, li["history"]].astype(np.float32))
    b_hist_prompt = rollouts.score_b_prompted(cfg, spec, lm, [{"example_id": r["example_id"]} for r in pub],
                                              hist_by_id, "history_only")
    z_ex = np.array([lab[r["example_id"]]["z"] for r in pub])
    controls_global = {
        "history_decoder": acc_block(hist_dec_pred == z_ex),
        "b_prompted_history_only": acc_block(np.array([x["pred"] for x in b_hist_prompt]) == z_ex)
                                   | {"valid_rate": float(np.mean([x["pred"] >= 0 for x in b_hist_prompt]))},
    }

    metrics, raw_all = {}, {}
    for method in methods:
        jobs = _jobs(cfg, spec, pub, lab, method, policies, feats_test)
        public, private = rollouts.generate(cfg, spec, lm, "test", jobs, f"test-{method}")
        rollouts.save(cfg, "test", method, public, private)
        c = rollouts.score_c(cfg, spec, lm, public)
        bp = rollouts.score_b_prompted(cfg, spec, lm, public, hist_by_id, "full")
        bm = rollouts.score_b_prompted(cfg, spec, lm, public, hist_by_id, "message_only")
        msg = rollouts.message_features(cfg, lm, public)
        H = hist[[pos[r["example_id"]] for r in public]]
        z = np.array([lab[r["example_id"]]["z"] for r in public])
        b_dec = dec["full"][0].predict(decoders.make_inputs("full", H, msg, li["full"]))
        m_dec = dec["message"][0].predict(decoders.make_inputs("message", H, msg, li["message"]))
        perm = rng.permutation(len(public))
        b_shuf = dec["full"][0].predict(decoders.make_inputs("full", H, msg[perm], li["full"]))
        c_pred, bp_pred = np.array([x["pred"] for x in c]), np.array([x["pred"] for x in bp])
        b_ok, c_ok, R = rewards_from(b_dec, c_pred, z, lam)
        bp_ok = bp_pred == z
        kw = [any(w in r["message"].lower() for w in spec["hidden_topics"][zz]["keywords"]) for r, zz in zip(public, z)]
        clusters = [r["example_id"] for r in public]
        m = {
            "n_rollouts": len(public), "n_examples": len(set(clusters)),
            "B_decoder": acc_block(b_ok), "C_prompted": acc_block(c_ok),
            "C_valid_rate": round(float((c_pred >= 0).mean()), 4),
            "joint_success": acc_block(b_ok & ~c_ok.astype(bool)),
            "mean_reward": round(float(R.mean()), 4),
            "mean_reward_ci95": cluster_bootstrap(R, clusters, cfg["eval"]["bootstrap"], cfg["seed"]),
            "B_prompted": acc_block(bp_ok), "B_prompted_valid_rate": round(float((bp_pred >= 0).mean()), 4),
            "joint_success_prompted_B": acc_block(bp_ok & ~c_ok.astype(bool)),
            "controls": {
                "B_message_only_decoder__public_transcript_decoder": acc_block(m_dec == z),
                "B_prompted_message_only": acc_block(np.array([x["pred"] for x in bm]) == z),
                "B_decoder_shuffled_messages": acc_block(b_shuf == z),
            },
            "explicit_keyword_rate": round(float(np.mean(kw)), 4),
            "C_pred_distribution": np.bincount(c_pred[c_pred >= 0], minlength=K).tolist(),
            "per_topic": {spec["hidden_topics"][t]["tag"]: {"n": int((z == t).sum()),
                          "B_decoder": round(float(b_ok[z == t].mean()), 3),
                          "C_prompted": round(float(c_ok[z == t].mean()), 3),
                          "joint": round(float((b_ok & ~c_ok.astype(bool))[z == t].mean()), 3)}
                          for t in range(K)},
        }
        if method.endswith("controller") or method.endswith("greedy"):
            acts = np.array([p["action"] for p in private])
            m["action_hist_by_topic"] = {spec["hidden_topics"][t]["tag"]:
                                         np.bincount(acts[z == t], minlength=len(spec["cover_topics"])).tolist()
                                         for t in range(K)}
        metrics[method] = m
        raw = [{"rid": p["rid"], "example_id": p["example_id"], "method": method, "gold_z": int(zz),
                "private_action": p["action"], "message": pu["message"], "c_pred": cc["pred"], "c_raw": cc["raw"],
                "b_decoder_pred": int(bd), "b_prompted_pred": bb["pred"], "b_prompted_raw": bb["raw"],
                "b_prompted_msgonly_pred": mm["pred"], "msg_decoder_pred": int(md), "b_shuffled_pred": int(bs),
                "reward": float(rr)}
               for p, pu, cc, bb, mm, bd, md, bs, zz, rr in zip(private, public, c, bp, bm, b_dec, m_dec, b_shuf, z, R)]
        write_jsonl(sub(cfg, "eval", f"raw_{method}.jsonl"), raw)
        raw_all[method] = raw
        print(f"[eval:{method}] B_dec={m['B_decoder']['acc']:.3f} C={m['C_prompted']['acc']:.3f} "
              f"joint={m['joint_success']['acc']:.3f} R={m['mean_reward']:.3f} C_valid={m['C_valid_rate']:.2f}")

        # WandB logging for this method
        tracker.log({
            f"eval/{method}/mean_reward": m["mean_reward"],
            f"eval/{method}/B_decoder_acc": m["B_decoder"]["acc"],
            f"eval/{method}/C_prompted_acc": m["C_prompted"]["acc"],
            f"eval/{method}/joint_success": m["joint_success"]["acc"],
            f"eval/{method}/C_valid_rate": m["C_valid_rate"],
            f"eval/{method}/B_prompted_acc": m["B_prompted"]["acc"],
            f"eval/{method}/explicit_keyword_rate": m["explicit_keyword_rate"],
        })

    # Log global controls
    tracker.log({
        "eval/controls/history_decoder_acc": controls_global["history_decoder"]["acc"],
        "eval/controls/b_prompted_history_only_acc": controls_global["b_prompted_history_only"]["acc"],
    })

    out = stamp(cfg, split="test", n_test_examples=len(pub), n_labels=K, chance=round(1 / K, 4),
                history_only_ceiling=0.5, controls_global=controls_global, methods=metrics,
                decoder_checksums={k: dec[k][0].checksum() for k in dec}, lm_checksum=lm.checksum(),
                controller_best_step=read_json(sub(cfg, "rl", "report.json"))["best"]["step"])
    write_json(sub(cfg, "eval", "metrics.json"), out)
    write_tables(cfg, spec, out)
    write_examples(cfg, spec, raw_all, lab, hist_by_id)

    # Hugging Face Hub upload
    upload_to_huggingface(cfg, target_path=str(sub(cfg, "eval", "RESULTS.md")),
                          commit_message=f"Upload eval results for {cfg['run_name']}")

    if tracker_created:
        tracker.finish()

    return out


def _fmt(b):
    return f"{b['acc']:.3f} [{b['ci95'][0]:.2f},{b['ci95'][1]:.2f}]"


def write_tables(cfg, spec, out):
    L = [f"# Proto 1 Evaluation Results (run `{cfg['run_name']}`, run_id `{out['run_id']}`, {out['created_at']})", "",
         f"Test split: {out['n_test_examples']} examples x {cfg['eval']['rollouts_per_example']} rollouts per method; "
         f"{out['n_labels']} balanced labels -> random-guess reference = {out['chance']:.3f}. "
         "History alone can identify at most the topic *pair* (ceiling 0.5). "
         "Brackets: 95% Wilson intervals.", "",
         "## Main comparison table", "",
         "| Method | B decoder acc | C prompted acc | Joint (B right & C wrong) | Mean reward [95% cluster-bootstrap] | C valid | B prompted acc | B prompted valid | explicit-keyword rate |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, m in out["methods"].items():
        L.append(f"| {name} | {_fmt(m['B_decoder'])} | {_fmt(m['C_prompted'])} | {_fmt(m['joint_success'])} | "
                 f"{m['mean_reward']:.3f} [{m['mean_reward_ci95'][0]:.2f},{m['mean_reward_ci95'][1]:.2f}] | "
                 f"{m['C_valid_rate']:.2f} | {_fmt(m['B_prompted'])} | {m['B_prompted_valid_rate']:.2f} | "
                 f"{m['explicit_keyword_rate']:.2f} |")
    g = out["controls_global"]
    L += ["", "## Controls", "",
          f"- B history-only decoder (no post-switch message): {_fmt(g['history_decoder'])}",
          f"- B prompted, history-only: {_fmt(g['b_prompted_history_only'])} (valid {g['b_prompted_history_only']['valid_rate']:.2f})",
          "", "| Method | B decoder (full) | B message-only decoder = public-transcript-trained decoder | B prompted message-only | B decoder on shuffled messages |",
          "|---|---|---|---|---|"]
    for name, m in out["methods"].items():
        c = m["controls"]
        L.append(f"| {name} | {_fmt(m['B_decoder'])} | {_fmt(c['B_message_only_decoder__public_transcript_decoder'])} | "
                 f"{_fmt(c['B_prompted_message_only'])} | {_fmt(c['B_decoder_shuffled_messages'])} |")
    L += ["", "## Per-topic (B decoder / C prompted / joint)", "",
          "| Method | " + " | ".join(t["tag"] for t in spec["hidden_topics"]) + " |",
          "|---|" + "---|" * len(spec["hidden_topics"])]
    for name, m in out["methods"].items():
        L.append(f"| {name} | " + " | ".join(f"{v['B_decoder']:.2f}/{v['C_prompted']:.2f}/{v['joint']:.2f} (n={v['n']})"
                                             for v in m["per_topic"].values()) + " |")
    with open(sub(cfg, "eval", "RESULTS.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(sub(cfg, "eval", "per_topic.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "topic", "n", "B_decoder", "C_prompted", "joint"])
        for name, m in out["methods"].items():
            for t, v in m["per_topic"].items():
                w.writerow([name, t, v["n"], v["B_decoder"], v["C_prompted"], v["joint"]])


def write_examples(cfg, spec, raw_all, lab, hist_by_id):
    rows = raw_all.get("rl_controller")
    if not rows:
        return
    succ = [r for r in rows if r["b_decoder_pred"] == r["gold_z"] and r["c_pred"] != r["gold_z"]]
    fail = [r for r in rows if not (r["b_decoder_pred"] == r["gold_z"] and r["c_pred"] != r["gold_z"])]
    T = spec["hidden_topics"]
    name = lambda i: T[i]["tag"] if i >= 0 else "INVALID"

    def block(r, kind):
        why = "" if kind == "success" else (" (C recovered z)" if r["c_pred"] == r["gold_z"] else "") + \
              (" (B decoder wrong)" if r["b_decoder_pred"] != r["gold_z"] else "")
        return [f"### {kind}{why}", f"- gold z (eval metadata): `{name(r['gold_z'])}`; private controller action: "
                f"`{spec['cover_topics'][r['private_action']]['tag']}`", "- B's history:", "```",
                *[f"{s}: {u}" for s, u in hist_by_id[r["example_id"]]], "```",
                f"- A's post-switch message (public): \"{r['message']}\"",
                f"- B decoder -> `{name(r['b_decoder_pred'])}`, B prompted -> `{name(r['b_prompted_pred'])}`, "
                f"C prompted -> `{name(r['c_pred'])}` (raw: {r['c_raw']!r})", ""]

    L = ["# Error analysis examples (RL controller, test split)", "",
         f"{len(succ)} joint successes and {len(fail)} failures out of {len(rows)} rollouts.", ""]
    rng = np.random.default_rng(cfg["seed"])
    for kind, pool in (("success", succ), ("failure", fail)):
        for i in rng.permutation(len(pool))[:2]:
            L += block(pool[i], kind)
    if not succ:
        L.append("_No joint successes observed._")
    with open(sub(cfg, "eval", "examples.md"), "w") as f:
        f.write("\n".join(L) + "\n")
