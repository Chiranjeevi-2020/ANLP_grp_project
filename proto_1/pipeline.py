"""All pipeline stages for proto_1 with WandB tracking and Hugging Face Hub checkpointing."""
import copy
import random

import numpy as np
import torch

from . import agents, decoders, rollouts
from .config import read_json, read_jsonl, stamp, sub, write_json, write_jsonl
from .controller import CoverController, MovingBaseline, reinforce_step
from .data import load_split
from .tracker import ExperimentTracker, upload_to_huggingface

SWITCHED = ("scripted", "prompt_only")


# --------------------------------------------------------------------------- decoder data
def gen_decoder_data(cfg, spec, lm):
    n = cfg["decoder"]["rollouts_per_example"]
    for split in ("train", "dev"):  # never test
        pub, _ = load_split(cfg, split)
        rollouts.history_features(cfg, lm, split)
        for method in cfg["decoder"]["methods"]:
            jobs = [{"example_id": r["example_id"], "method": method, "rollout": k} for r in pub for k in range(n)]
            public, private = rollouts.generate(cfg, spec, lm, split, jobs, f"dec-{split}-{method}")
            rollouts.save(cfg, split, f"dec_{method}", public, private)
            np.save(sub(cfg, "rollouts", split, f"dec_{method}.msgfeat.npy"), rollouts.message_features(cfg, lm, public))
    write_json(sub(cfg, "rollouts", "decoder_data_meta.json"),
               stamp(cfg, splits=["train", "dev"], methods=cfg["decoder"]["methods"], gen=cfg["gen"]))


def _hist_rows(cfg, split, public):
    hist, ids = rollouts.history_features(cfg, None, split)
    pos = {e: i for i, e in enumerate(ids)}
    return hist[[pos[r["example_id"]] for r in public]]


def _gather(cfg, split, names):
    _, lab = load_split(cfg, split)
    H, M, Y, meth = [], [], [], []
    for name in names:
        pub = rollouts.load_public(cfg, split, f"dec_{name}")
        H.append(_hist_rows(cfg, split, pub))
        M.append(np.load(sub(cfg, "rollouts", split, f"dec_{name}.msgfeat.npy")))
        Y += [lab[r["example_id"]]["z"] for r in pub]
        meth += [name] * len(pub)
    return {"hist": np.concatenate(H), "msg": np.concatenate(M), "y": np.array(Y), "method": np.array(meth)}


def train_decoders(cfg, spec):
    names = cfg["decoder"]["methods"]
    tr, dev = _gather(cfg, "train", names), _gather(cfg, "dev", names)
    dev["sel"] = np.isin(dev["method"], SWITCHED)  # select on switched dev rollouts (the RL regime)
    K, layers = len(spec["hidden_topics"]), cfg["features"]["layers"]
    report = {"train_rows": int(len(tr["y"])), "dev_rows": int(len(dev["y"])), "kinds": {}}
    for kind in decoders.KINDS:
        model, info = decoders.select_and_train(kind, tr, dev, K, layers, cfg)
        li = info["best"]["layer_idx"]
        pred = model.predict(decoders.make_inputs(kind, dev["hist"], dev["msg"], li))
        info["dev_acc_by_method"] = {m: float((pred[dev["method"] == m] == dev["y"][dev["method"] == m]).mean())
                                     for m in names}
        info["dev_acc_switched"] = float((pred[dev["sel"]] == dev["y"][dev["sel"]]).mean())
        info.update(stamp(cfg, trained_on="train", selected_on="dev"))
        decoders.save(sub(cfg, "decoders", f"{kind}.pt"), model, info)
        report["kinds"][kind] = {k: v for k, v in info.items() if k != "grid"} | {"grid": info["grid"]}
        print(f"[decoder:{kind}] layer={info['best']['layer']} wd={info['best']['weight_decay']} "
              f"dev_switched={info['dev_acc_switched']:.3f} by_method={info['dev_acc_by_method']}")
    full, hist = report["kinds"]["full"]["dev_acc_switched"], report["kinds"]["history"]["dev_acc_switched"]
    d = cfg["decoder"]
    report["reward_gate"] = {"full_dev_switched": full, "history_dev_switched": hist, "chance": 1 / K,
                             "min_dev_acc": d["min_dev_acc"], "min_gain_over_history": d["min_gain_over_history"],
                             "passed": bool(full >= d["min_dev_acc"] and full - hist >= d["min_gain_over_history"])}
    report.update(stamp(cfg))
    write_json(sub(cfg, "decoders", "report.json"), report)
    print(f"[decoder] reward gate passed={report['reward_gate']['passed']} (full {full:.3f} vs history {hist:.3f})")
    if not report["reward_gate"]["passed"]:
        print("[decoder] WARNING: B decoder is not clearly better than history-only on dev. RL will still run "
              "but its reward is weak; this is recorded in the audit.")
    return report


def _full_decoder(cfg):
    m, info = decoders.load(sub(cfg, "decoders", "full.pt"))
    return m, info["best"]["layer_idx"]


def b_decoder_predict(cfg, split, public, msg_feats, model, li):
    return model.predict(decoders.make_inputs("full", _hist_rows(cfg, split, public), msg_feats, li))


def rewards_from(b_pred, c_pred, z, lam):
    b_ok, c_ok = (np.asarray(b_pred) == np.asarray(z)), (np.asarray(c_pred) == np.asarray(z))
    return b_ok.astype(int), c_ok.astype(int), b_ok.astype(float) - lam * c_ok.astype(float)


# --------------------------------------------------------------------------- rollout bank
def build_bank(cfg, spec, lm):
    """Pre-generates A's message for every (train/dev state, action, k). RL samples from it."""
    model, li = _full_decoder(cfg)
    A, lam = len(spec["cover_topics"]), cfg["rl"]["lambda"]
    for split, k in (("train", cfg["bank"]["k_train"]), ("dev", cfg["bank"]["k_dev"])):
        pub, lab = load_split(cfg, split)
        jobs = [{"example_id": r["example_id"], "method": "controller", "action": a, "rollout": j}
                for r in pub for a in range(A) for j in range(k)]
        public, private = rollouts.generate(cfg, spec, lm, split, jobs, f"bank-{split}")
        rollouts.save(cfg, split, "bank", public, private)
        c = rollouts.score_c(cfg, spec, lm, public)
        msg = rollouts.message_features(cfg, lm, public)
        np.save(sub(cfg, "rollouts", split, "bank.msgfeat.npy"), msg)
        b = b_decoder_predict(cfg, split, public, msg, model, li)
        z = [lab[r["example_id"]]["z"] for r in public]
        b_ok, c_ok, R = rewards_from(b, [x["pred"] for x in c], z, lam)
        rows = [{"rid": p["rid"], "example_id": p["example_id"], "action": p["action"], "rollout": p["rollout"],
                 "b_pred": int(bp), "c_pred": cc["pred"], "c_raw": cc["raw"], "b_correct": int(bo),
                 "c_correct": int(co), "reward": float(r)}
                for p, bp, cc, bo, co, r in zip(private, b, c, b_ok, c_ok, R)]
        write_jsonl(sub(cfg, "rollouts", split, "bank.scores.jsonl"), rows)
        print(f"[bank:{split}] n={len(rows)} B={b_ok.mean():.3f} C={c_ok.mean():.3f} "
              f"C_valid={np.mean([x['pred'] >= 0 for x in c]):.3f} R={R.mean():.3f}")
    write_json(sub(cfg, "rollouts", "bank_meta.json"), stamp(cfg, splits=["train", "dev"], gen=cfg["gen"]))


def _bank_table(cfg, spec, split):
    pub, lab = load_split(cfg, split)
    pos = {r["example_id"]: i for i, r in enumerate(pub)}
    A, k = len(spec["cover_topics"]), cfg["bank"]["k_train" if split == "train" else "k_dev"]
    R, B, C = (np.full((len(pub), A, k), np.nan) for _ in range(3))
    for r in read_jsonl(sub(cfg, "rollouts", split, "bank.scores.jsonl")):
        i = pos[r["example_id"]]
        R[i, r["action"], r["rollout"]], B[i, r["action"], r["rollout"]], C[i, r["action"], r["rollout"]] = \
            r["reward"], r["b_correct"], r["c_correct"]
    assert not np.isnan(R).any(), f"incomplete bank for {split}"
    z = np.array([lab[r["example_id"]]["z"] for r in pub])
    return R, B, C, z


def _policy_inputs(cfg, split, li):
    hist, _ = rollouts.history_features(cfg, None, split)
    return torch.as_tensor(hist[:, li].astype(np.float32))


def new_policy(cfg, spec, feat_train):
    c = cfg["controller"]
    torch.manual_seed(cfg["seed"])
    return CoverController(feat_train.shape[1], len(spec["hidden_topics"]), len(spec["cover_topics"]),
                           c["topic_emb_dim"], c["hidden_dim"], c["init_std"],
                           feat_train.mean(0), feat_train.std(0) + 1e-4)


def save_policy(path, policy, extra):
    torch.save({"state_dict": policy.state_dict(), "extra": extra}, path)


def load_policy(cfg, spec, path, feat_train):
    p = new_policy(cfg, spec, feat_train)
    p.load_state_dict(torch.load(path, weights_only=False)["state_dict"])
    return p.eval()


@torch.no_grad()
def dev_expected(policy, feats, z, R, B, C):
    """Exact expectation over the policy's action distribution using the dev bank (no sampling noise)."""
    policy.eval()
    probs = policy.dist(feats, torch.as_tensor(z)).probs.numpy()
    f = lambda T: float((probs * T.mean(-1)).sum(1).mean())
    ent = float(-(probs * np.log(probs + 1e-12)).sum(1).mean())
    return {"reward": f(R), "b_acc": f(B), "c_acc": f(C), "entropy": ent}


# --------------------------------------------------------------------------- smoke test
def smoke(cfg, spec, lm):
    model, li = _full_decoder(cfg)
    pub, lab = load_split(cfg, "train")
    feats = _policy_inputs(cfg, "train", li)
    policy = new_policy(cfg, spec, feats)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg["rl"]["lr"])
    baseline = MovingBaseline(cfg["rl"]["baseline_momentum"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    lm_before, dec_before = lm.checksum(), model.checksum()
    w_before = copy.deepcopy(policy.state_dict())
    steps = []
    for s in range(cfg["smoke"]["steps"]):
        idx = rng.sample(range(len(pub)), cfg["smoke"]["batch_size"])
        z = torch.as_tensor([lab[pub[i]["example_id"]]["z"] for i in idx])

        def reward_fn(actions):
            jobs = [{"example_id": pub[i]["example_id"], "method": "controller", "action": int(a), "rollout": s}
                    for i, a in zip(idx, actions)]
            public, _ = rollouts.generate(cfg, spec, lm, "train", jobs, f"smoke-{s}")
            c = rollouts.score_c(cfg, spec, lm, public)
            b = b_decoder_predict(cfg, "train", public, rollouts.message_features(cfg, lm, public), model, li)
            return rewards_from(b, [x["pred"] for x in c], z.numpy(), cfg["rl"]["lambda"])[2]

        st = reinforce_step(policy, opt, feats[idx], z, reward_fn, baseline, cfg["rl"], gen)
        steps.append(st)
        print(f"[smoke] step {s} R={st['reward_mean']:.2f} grad={st['grad_norm']:.4g} adv={st['adv_abs_mean']:.3f}")
    w_after = policy.state_dict()
    changed = any(not torch.equal(w_before[k], w_after[k]) for k in w_before if not k.startswith("feat_"))
    signal = [s for s in steps if s["adv_abs_mean"] > 0]
    with torch.no_grad():
        probs = policy.dist(feats, torch.as_tensor([lab[r["example_id"]]["z"] for r in pub])).probs
    checks = {
        "grads_finite": all(s["grad_finite"] for s in steps),
        "grads_nonzero_when_signal": bool(signal) and all(s["grad_norm"] > 0 for s in signal),
        "had_learning_signal": bool(signal),
        "controller_weights_changed": bool(changed),
        "frozen_lm_unchanged": lm.checksum() == lm_before and not lm.any_requires_grad(),
        "b_decoder_unchanged": model.checksum() == dec_before and not any(p.requires_grad for p in model.parameters()),
        "action_probs_valid": bool(torch.isfinite(probs).all() and (probs >= 0).all()
                                   and torch.allclose(probs.sum(-1), torch.ones(len(probs)), atol=1e-4)),
        "rewards_in_set": all(set(s["rewards"]) <= {-1.0, 0.0, 1.0} for s in steps),
        "updates_from_train_split_only": True,
    }
    out = stamp(cfg, checks=checks, passed=all(checks.values()), steps=steps, lm_checksum=lm_before,
                decoder_checksum=dec_before)
    write_json(sub(cfg, "rl", "smoke.json"), out)
    print(f"[smoke] passed={out['passed']} {checks}")
    return out


# --------------------------------------------------------------------------- REINFORCE on the bank
def train_rl(cfg, spec, tracker=None):
    smoke_res = read_json(sub(cfg, "rl", "smoke.json"))
    if not smoke_res["passed"]:
        raise RuntimeError("smoke test failed; refusing to run RL")

    tracker_created = False
    if tracker is None:
        tracker = ExperimentTracker(cfg, job_type="train-rl")
        tracker_created = True

    _, li = _full_decoder(cfg)
    Rtr, Btr, Ctr, ztr = _bank_table(cfg, spec, "train")
    Rdv, Bdv, Cdv, zdv = _bank_table(cfg, spec, "dev")
    ftr, fdv = _policy_inputs(cfg, "train", li), _policy_inputs(cfg, "dev", li)
    rl = cfg["rl"]
    policy = new_policy(cfg, spec, ftr)
    save_policy(sub(cfg, "rl", "controller_init.pt"), policy, stamp(cfg, step=0))
    opt = torch.optim.Adam(policy.parameters(), lr=rl["lr"])
    baseline = MovingBaseline(rl["baseline_momentum"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    k = Rtr.shape[2]
    init_dev = dev_expected(policy, fdv, zdv, Rdv, Bdv, Cdv)
    curve, best = [{"step": 0, "dev": init_dev}], {"step": 0, **init_dev}
    save_policy(sub(cfg, "rl", "controller_best.pt"), policy, stamp(cfg, step=0, dev=init_dev))
    print(f"[rl] step 0 dev={init_dev}")

    tracker.log({
        "rl/step": 0,
        "rl/dev_reward": init_dev["reward"],
        "rl/dev_b_acc": init_dev["b_acc"],
        "rl/dev_c_acc": init_dev["c_acc"],
        "rl/dev_entropy": init_dev["entropy"],
    }, step=0)

    for step in range(1, rl["steps"] + 1):
        idx = rng.integers(0, len(ztr), rl["batch_size"])
        ks = rng.integers(0, k, rl["batch_size"])
        reward_fn = lambda a: Rtr[idx, a.numpy(), ks]
        st = reinforce_step(policy, opt, ftr[idx], torch.as_tensor(ztr[idx]), reward_fn, baseline, rl, gen)
        assert st["grad_finite"], f"non-finite gradient at step {step}"
        rec = {"step": step, **{x: st[x] for x in ("loss", "reward_mean", "baseline_before", "entropy", "grad_norm")}}

        step_log = {
            "rl/step": step,
            "rl/train_loss": st["loss"],
            "rl/train_reward": st["reward_mean"],
            "rl/baseline": st["baseline_before"],
            "rl/entropy": st["entropy"],
            "rl/grad_norm": st["grad_norm"],
        }

        if step % rl["eval_every"] == 0 or step == rl["steps"]:
            rec["dev"] = dev_expected(policy, fdv, zdv, Rdv, Bdv, Cdv)
            if rec["dev"]["reward"] > best["reward"] + 1e-9:
                best = {"step": step, **rec["dev"]}
                save_policy(sub(cfg, "rl", "controller_best.pt"), policy, stamp(cfg, step=step, dev=rec["dev"]))
            print(f"[rl] step {step} train_R={st['reward_mean']:.3f} base={st['baseline_before']:.3f} "
                  f"ent={st['entropy']:.3f} dev={rec['dev']}")
            step_log.update({
                "rl/dev_reward": rec["dev"]["reward"],
                "rl/dev_b_acc": rec["dev"]["b_acc"],
                "rl/dev_c_acc": rec["dev"]["c_acc"],
                "rl/dev_entropy": rec["dev"]["entropy"],
                "rl/best_dev_reward": best["reward"],
            })

        tracker.log(step_log, step=step)
        curve.append(rec)

    per_topic_best = {int(t): int(np.nanmean(Rtr[ztr == t], axis=(0, 2)).argmax()) for t in np.unique(ztr)}
    oracle_dev = float(np.mean([Rdv[i, per_topic_best[int(zdv[i])]].mean() for i in range(len(zdv))]))
    report = stamp(cfg, best=best, init_dev=init_dev, steps=rl["steps"], trained_on="train", selected_on="dev",
                   uniform_policy_dev_reward=float(Rdv.mean()), train_bank_mean_reward=float(Rtr.mean()),
                   per_topic_best_fixed_action_dev_reward=oracle_dev, per_topic_best_fixed_action=per_topic_best,
                   final_baseline=baseline.value)
    write_json(sub(cfg, "rl", "curve.json"), curve)
    write_json(sub(cfg, "rl", "report.json"), report)
    plot_curve(cfg, curve, report)
    tracker.log_image("rl/training_curve", str(sub(cfg, "rl", "training_curve.png")))
    tracker.log_summary({
        "best_dev_step": best["step"],
        "best_dev_reward": best["reward"],
        "best_dev_b_acc": best["b_acc"],
        "best_dev_c_acc": best["c_acc"],
        "best_dev_entropy": best["entropy"],
    })

    print(f"[rl] best dev step={best['step']} reward={best['reward']:.3f} (init {init_dev['reward']:.3f}, "
          f"uniform {report['uniform_policy_dev_reward']:.3f})")

    # Optional Hugging Face Hub upload
    best_ckpt = sub(cfg, "rl", "controller_best.pt")
    upload_to_huggingface(cfg, target_path=str(best_ckpt), commit_message=f"Upload controller_best for {cfg['run_name']}")

    if tracker_created:
        tracker.finish()

    return report


def plot_curve(cfg, curve, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = [c["step"] for c in curve if "reward_mean" in c]
    r = np.array([c["reward_mean"] for c in curve if "reward_mean" in c])
    w = max(1, len(r) // 30)
    sm = np.convolve(r, np.ones(w) / w, mode="valid")
    ev = [c for c in curve if "dev" in c]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    ax[0].plot(steps[w - 1:], sm, label=f"train batch reward (MA{w})")
    ax[0].plot([c["step"] for c in ev], [c["dev"]["reward"] for c in ev], "o-", label="dev expected reward")
    ax[0].axhline(report["uniform_policy_dev_reward"], ls="--", c="gray", label="uniform policy (dev)")
    ax[0].axvline(report["best"]["step"], ls=":", c="red", label="selected ckpt")
    ax[0].set_xlabel("step"); ax[0].legend(fontsize=7); ax[0].set_title("Reward (lambda=1)")
    ax[1].plot([c["step"] for c in ev], [c["dev"]["b_acc"] for c in ev], label="B (decoder)")
    ax[1].plot([c["step"] for c in ev], [c["dev"]["c_acc"] for c in ev], label="C (prompted)")
    ax[1].set_xlabel("step"); ax[1].legend(fontsize=7); ax[1].set_title("Dev expected accuracy")
    ax[2].plot([c["step"] for c in ev], [c["dev"]["entropy"] for c in ev])
    ax[2].set_xlabel("step"); ax[2].set_title("Dev policy entropy (nats)")
    fig.tight_layout()
    fig.savefig(sub(cfg, "rl", "training_curve.png"), dpi=130)
    plt.close(fig)
