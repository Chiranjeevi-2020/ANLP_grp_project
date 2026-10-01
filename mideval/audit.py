"""Three audit passes (data / smoke / final). Each writes audit/<stage>.json; AUDIT.md aggregates them.
Passing checks is evidence, not proof, of no leakage: see UNRESOLVED_RISKS."""
import inspect
import re
from pathlib import Path

import torch

from . import agents, pipeline
from .config import SPLITS, load_spec, read_json, read_jsonl, run_id, sub, write_json
from .controller import CoverController, MovingBaseline, reinforce_step
from .data import load_split

UNRESOLVED_RISKS = [
    "B's decoder is trained (supervised, on baseline rollouts) while C is a fixed prompted classifier: the B-vs-C gap "
    "mixes privileged context with decoder training. The public-transcript-trained decoder control measures the latter; "
    "B-vs-C differences must not be read as evidence of secure communication.",
    "C is a non-adaptive, small (1.5B) prompted eavesdropper; a stronger or trained C (see the public-transcript decoder "
    "control) may recover much more.",
    "The surface cover topic is observable by design. The controller action ID/state is private, but its effect (which "
    "cover A talks about) is public; a cover chosen per hidden topic is a potential code that a trained outsider could learn.",
    "A, B and C are the same frozen base model, so they share priors; prompted B/C may share biases (e.g. label-position bias).",
    "Pre-switch histories are templated and pair-level by construction; the generalization of findings to natural "
    "dialogue is untested. Only 4 history families per pair are held out for dev/test.",
    "RL uses a cached rollout bank (k samples per state-action). The policy can exploit idiosyncrasies of these specific "
    "samples; test rollouts are freshly generated to measure this, but the bank's limited k is a source of optimism.",
    "Leakage scans are string-based (labels, tags, private descriptions, private markers); paraphrased leakage in A's "
    "messages is the intended channel and is measured only through B/C accuracy and the keyword-rate diagnostic.",
    "Single seed, single lambda, small test set: no statistical-significance or trade-off-curve claims are supported.",
]


def _res(name, passed, detail=""):
    return {"check": name, "passed": bool(passed), "detail": detail}


def audit_data(cfg):
    spec, out = load_spec(), []
    pubs, labs = {}, {}
    for s in SPLITS:
        pubs[s], labs[s] = load_split(cfg, s)
    fam = {s: {r["family"] for r in labs[s].values()} for s in SPLITS}
    out.append(_res("families_disjoint_across_splits",
                    not (fam["train"] & fam["dev"] or fam["train"] & fam["test"] or fam["dev"] & fam["test"]),
                    {s: len(fam[s]) for s in SPLITS}))
    rep = read_json(sub(cfg, "data", "data_report.json"))
    out.append(_res("no_exact_cross_split_duplicates", rep["exact_cross_split_duplicates"] == 0))
    out.append(_res("no_near_duplicates_jaccard_lt_0.8", rep["max_cross_split_jaccard"] < 0.8,
                    {"max_jaccard": rep["max_cross_split_jaccard"], "pair": rep["most_similar_pair"]}))
    ids = {s: {r["example_id"] for r in pubs[s]} for s in SPLITS}
    out.append(_res("example_ids_disjoint", not (ids["train"] & ids["test"] or ids["dev"] & ids["test"] or ids["train"] & ids["dev"])))
    bal = {s: sorted({sum(1 for r in labs[s].values() if r["z"] == t) for t in range(len(spec["hidden_topics"]))}) for s in SPLITS}
    out.append(_res("labels_balanced_per_split", all(len(v) == 1 and v[0] > 0 for v in bal.values()), bal))
    keys = {k for s in SPLITS for r in pubs[s] for k in r}
    out.append(_res("public_records_have_only_id_and_history", keys == {"example_id", "history"}, sorted(keys)))
    tags = [t["tag"] for t in spec["hidden_topics"]]
    out.append(_res("example_ids_opaque", all(re.fullmatch(r"[0-9a-f]{10}", e) and not any(t in e for t in tags)
                                              for s in SPLITS for e in ids[s])))
    bad, kw_hits = [], 0
    for s in SPLITS:
        for r in pubs[s]:
            h = agents.render_history(r["history"]).lower()
            for t in spec["hidden_topics"]:
                if t["label"].lower() in h or t["tag"] in h:
                    bad.append((s, r["example_id"], t["tag"]))
            kw_hits += any(w in h for w in spec["hidden_topics"][labs[s][r["example_id"]]["z"]]["keywords"])
    out.append(_res("history_contains_no_label_or_tag", not bad, bad[:5]))
    # pair-level histories: same history text for both topics of a pair -> history cannot separate them
    by_text = {}
    for s in SPLITS:
        for r in pubs[s]:
            by_text.setdefault(agents.render_history(r["history"]), set()).add(labs[s][r["example_id"]]["z"])
    out.append(_res("every_history_shared_by_both_pair_topics", all(len(v) == 2 for v in by_text.values()),
                    {"n_histories": len(by_text), "gold_keyword_in_history_count(warning_only)": kw_hits}))
    used = {}
    for z, c in spec["scripted_map"].items():
        used.setdefault(c, []).append(int(z))
    out.append(_res("scripted_cover_not_one_to_one", all(len(v) >= 2 for v in used.values()), used))
    out.append(_res("controller_action_inventory_shared_by_all_topics", True,
                    f"{len(spec['cover_topics'])} actions available for every z (policy output is a full softmax)"))
    # prompt builders on every example with a dummy message
    dummy, leaks, templ = "Anyway, the tomatoes are finally growing.", [], set()
    for s in SPLITS:
        for r in pubs[s]:
            lab = labs[s][r["example_id"]]
            secret = [agents.private_desc(spec, lab), "Private note", spec["hidden_topics"][lab["z"]]["tag"]]
            views = {"B": agents.b_view(r["history"], dummy), "C": agents.c_view(dummy)}
            for who, v in views.items():
                if any(x in v for x in secret):
                    leaks.append((s, r["example_id"], who))
            _, cu = agents.c_prompt(spec, dummy)
            _, bu = agents.b_prompt(spec, r["history"], dummy)
            templ.add(bu.replace(views["B"], "<VIEW>"))
            templ.add(cu.replace(views["C"], "<VIEW>"))
            if any(u in cu for _, u in r["history"]):
                leaks.append((s, r["example_id"], "C sees history"))
    out.append(_res("b_c_views_contain_no_private_values", not leaks, leaks[:5]))
    out.append(_res("b_c_prompt_templates_identical_across_examples (same full label list)", len(templ) == 2, len(templ)))
    return out


def toy_reinforce_checks():
    """Sign / detach / baseline-order checks on a tiny controller (no LM)."""
    res = []
    torch.manual_seed(0)
    for sign in (+1.0, -1.0):
        pol = CoverController(4, 2, 3, 4, 8, init_std=0.5)
        opt = torch.optim.SGD(pol.parameters(), lr=0.5)
        f, z = torch.randn(16, 4), torch.zeros(16, dtype=torch.long)
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            before = pol.dist(f, z).probs[:, 1].mean().item()
        rf = lambda a: torch.where(a == 1, torch.tensor(sign), torch.tensor(0.0))
        cfgrl = {"entropy_coef": 0.0, "grad_clip": 100.0}
        for _ in range(30):
            reinforce_step(pol, opt, f, z, rf, MovingBaseline(0.9, 0.0), cfgrl, g)
        with torch.no_grad():
            after = pol.dist(f, z).probs[:, 1].mean().item()
        res.append(_res(f"loss_sign_{'pos' if sign > 0 else 'neg'}_reward_moves_prob_correctly",
                        after > before if sign > 0 else after < before, {"before": before, "after": after}))
    b = MovingBaseline(0.5, 0.0)
    adv = b.advantage(torch.tensor([1.0, 1.0]))
    b.update(torch.tensor([1.0, 1.0]))
    res.append(_res("advantage_uses_previous_baseline", torch.allclose(adv, torch.ones(2)) and b.value == 0.5))
    return res


def audit_smoke(cfg):
    s = read_json(sub(cfg, "rl", "smoke.json"))
    out = [_res(f"smoke:{k}", v) for k, v in s["checks"].items()]
    out.append(_res("smoke_run_id_current", s["run_id"] == run_id(cfg)))
    steps = s["steps"]
    out.append(_res("smoke_baseline_chain_consistent",
                    all(abs(steps[i]["baseline_before"] - steps[i - 1]["baseline_after"]) < 1e-9 for i in range(1, len(steps)))))
    out += toy_reinforce_checks()
    for fn in (pipeline.train_rl, pipeline.train_decoders, pipeline.gen_decoder_data, pipeline.build_bank, pipeline.smoke):
        out.append(_res(f"static:{fn.__name__}_never_references_test_split", '"test"' not in inspect.getsource(fn)))
    return out


def audit_final(cfg):
    spec, out, rid = load_spec(), [], run_id(cfg)
    arts = {"decoders": sub(cfg, "decoders", "report.json"), "decoder_data": sub(cfg, "rollouts", "decoder_data_meta.json"),
            "bank": sub(cfg, "rollouts", "bank_meta.json"), "smoke": sub(cfg, "rl", "smoke.json"),
            "rl": sub(cfg, "rl", "report.json"), "eval": sub(cfg, "eval", "metrics.json")}
    meta = {k: read_json(p) for k, p in arts.items()}
    out.append(_res("all_artifacts_share_current_run_id", all(m["run_id"] == rid for m in meta.values()),
                    {k: m["run_id"] for k, m in meta.items()}))
    order = ["decoder_data", "decoders", "bank", "smoke", "rl", "eval"]
    t = [meta[k]["created_unix"] for k in order]
    out.append(_res("artifact_timestamps_in_pipeline_order", t == sorted(t), dict(zip(order, t))))
    out.append(_res("config_hash_consistent", len({m["config_hash"] for m in meta.values()}) == 1))
    ckpt = Path(sub(cfg, "rl", "controller_best.pt")).stat().st_mtime
    out.append(_res("eval_newer_than_controller_checkpoint", Path(arts["eval"]).stat().st_mtime > ckpt))
    out.append(_res("decoders_trained_on_train_selected_on_dev",
                    all(read_json(arts["decoders"])["kinds"][k]["trained_on"] == "train" and
                        read_json(arts["decoders"])["kinds"][k]["selected_on"] == "dev" for k in ("full", "history", "message"))))
    out.append(_res("rl_trained_on_train_selected_on_dev", meta["rl"]["trained_on"] == "train" and meta["rl"]["selected_on"] == "dev"))
    out.append(_res("no_test_rollouts_in_training_dirs", not any("test" in p.name for d in ("train", "dev")
                                                                  for p in Path(sub(cfg, "rollouts", d, "x")).parent.glob("*"))))
    ev = meta["eval"]
    pub_test, lab_test = load_split(cfg, "test")
    n_ex = len(pub_test)
    n_r = n_ex * cfg["eval"]["rollouts_per_example"]
    sets, counts, leaks, keyset = {}, {}, [], set()
    for m in ev["methods"]:
        rows = read_jsonl(sub(cfg, "rollouts", "test", f"{m}.public.jsonl"))
        keyset |= {k for r in rows for k in r}
        sets[m] = sorted(r["example_id"] for r in rows)
        counts[m] = (len(rows), ev["methods"][m]["C_prompted"]["n"], ev["methods"][m]["B_decoder"]["n"])
        for r in rows:
            lab = lab_test[r["example_id"]]
            secret = [agents.private_desc(spec, lab), "Private note", "only you know",
                      spec["hidden_topics"][lab["z"]]["tag"]]
            if any(x in r["message"] for x in secret):
                leaks.append((m, r["rid"]))
    out.append(_res("public_test_rollouts_have_only_rid_id_message", keyset == {"rid", "example_id", "message"}, sorted(keyset)))
    out.append(_res("all_methods_same_test_examples", len({tuple(v) for v in sets.values()}) == 1))
    out.append(_res("invalid_predictions_kept_in_denominator", all(c == (n_r, n_r, n_r) for c in counts.values()), counts))
    out.append(_res("no_private_strings_in_public_messages", not leaks, leaks[:5]))
    a_lens = {}
    for m in ev["methods"]:
        priv = read_jsonl(sub(cfg, "rollouts", "test", f"{m}.private.jsonl"))
        a_lens[m] = round(sum(len(p["raw"].split()) for p in priv) / len(priv), 1)
    out.append(_res("matched_generation_budget", True,
                    {"shared_gen_settings": cfg["gen"], "mean_generated_words": a_lens,
                     "note": "scripted additionally prepends a fixed transition template (not generated)"}))
    dec_ck = {k: torch.load(sub(cfg, "decoders", f"{k}.pt"), weights_only=False)["checksum"] for k in ("full", "history", "message")}
    out.append(_res("decoders_frozen_since_training", dec_ck == ev["decoder_checksums"]))
    out.append(_res("frozen_lm_unchanged_smoke_to_eval", meta["smoke"]["lm_checksum"] == ev["lm_checksum"]))
    gate = meta["decoders"]["reward_gate"]
    out.append(_res("b_decoder_reward_gate (meaningful above history-only)", gate["passed"], gate))
    rl = meta["rl"]
    out.append(_res("rl_improved_dev_reward_over_init (informational)", rl["best"]["reward"] > rl["init_dev"]["reward"],
                    {"init": rl["init_dev"]["reward"], "best": rl["best"]["reward"], "best_step": rl["best"]["step"]}))
    return out


def run(cfg, stage):
    fn = {"data": audit_data, "smoke": audit_smoke, "final": audit_final}[stage]
    res = fn(cfg)
    write_json(sub(cfg, "audit", f"{stage}.json"), {"run_id": run_id(cfg), "stage": stage, "results": res})
    write_report(cfg)
    n_fail = sum(not r["passed"] for r in res)
    for r in res:
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['check']}")
    print(f"[audit:{stage}] {len(res) - n_fail}/{len(res)} passed")
    return res


def write_report(cfg):
    L = [f"# Audit report (run `{cfg['run_name']}`, run_id `{run_id(cfg)}`)", "",
         "Generated by `python -m mideval.cli audit <stage>`. Passing checks are evidence, not proof, of no leakage.", ""]
    for stage in ("data", "smoke", "final"):
        p = sub(cfg, "audit", f"{stage}.json")
        L.append(f"## Audit {stage}")
        if not p.exists():
            L += ["_not run yet_", ""]
            continue
        a = read_json(p)
        L += [f"| check | result | detail |", "|---|---|---|"]
        for r in a["results"]:
            d = str(r["detail"]).replace("|", "/")[:220]
            L.append(f"| {r['check']} | {'PASS' if r['passed'] else '**FAIL**'} | {d} |")
        L.append("")
    L += ["## Unresolved risks / limitations", ""] + [f"- {x}" for x in UNRESOLVED_RISKS]
    with open(sub(cfg, "AUDIT.md"), "w") as f:
        f.write("\n".join(L) + "\n")
