"""Unit tests for proto_1 (fast, no LM needed).

Run: pytest -q proto_1/tests
"""
import copy
import numpy as np
import pytest
import torch

from proto_1 import agents, decoders
from proto_1.config import load_config, load_spec
from proto_1.controller import CoverController, MovingBaseline, reinforce_loss, reinforce_step
from proto_1.data import build_examples, duplicate_report
from proto_1.evaluate import cluster_bootstrap, wilson
from proto_1.lm import resolve_device, resolve_dtype
from proto_1.pipeline import rewards_from
from proto_1.tracker import ExperimentTracker

SPEC = load_spec()
CFG = load_config("proto_1/configs/tiny.json")
K = len(SPEC["hidden_topics"])


@pytest.fixture(scope="module")
def data():
    return build_examples(CFG, SPEC)


def test_splits_family_disjoint_balanced_no_dups(data):
    pub, priv = data
    fams = {s: {r["family"] for r in priv[s]} for s in priv}
    assert not (fams["train"] & fams["test"]) and not (fams["dev"] & fams["test"]) and not (fams["train"] & fams["dev"])
    for s in priv:
        counts = np.bincount([r["z"] for r in priv[s]], minlength=K)
        assert counts.min() == counts.max() > 0
    rep = duplicate_report(pub)
    assert rep["exact_cross_split_duplicates"] == 0 and rep["max_cross_split_jaccard"] < 0.8


def test_public_records_carry_no_labels(data):
    pub, priv = data
    for s in pub:
        for r, p in zip(pub[s], priv[s]):
            assert set(r) == {"example_id", "history"} and r["example_id"] == p["example_id"]
            h = agents.render_history(r["history"])
            for t in SPEC["hidden_topics"]:
                assert t["label"] not in h and t["tag"] not in h


def test_history_is_pair_ambiguous(data):
    pub, priv = data
    seen = {}
    for s in pub:
        for r, p in zip(pub[s], priv[s]):
            seen.setdefault(agents.render_history(r["history"]), set()).add(p["z"])
    assert all(len(v) == 2 for v in seen.values())


def test_views_respect_boundaries(data):
    pub, priv = data
    r, p = pub["test"][0], priv["test"][0]
    msg = "Anyway, my basil plant keeps wilting."
    secret = agents.private_desc(SPEC, p)
    _, cu = agents.c_prompt(SPEC, msg)
    _, bu = agents.b_prompt(SPEC, r["history"], msg)
    for u in (cu, bu):
        assert secret not in u and "Private note" not in u and SPEC["hidden_topics"][p["z"]]["tag"] not in u
    assert all(utt not in cu for _, utt in r["history"])
    assert all(utt in bu for _, utt in r["history"])
    assert agents.label_list(SPEC) in cu and agents.label_list(SPEC) in bu
    _, au = agents.a_prompt(SPEC, "controller", r["history"], p, action=3, example_id=r["example_id"])[:2]
    assert secret in au and SPEC["cover_topics"][3]["label"] in au
    _, su, pre = agents.a_prompt(SPEC, "scripted", r["history"], p, example_id=r["example_id"])
    assert secret not in su and pre in su


def test_scripted_map_not_one_to_one():
    inv = {}
    for z, c in SPEC["scripted_map"].items():
        inv.setdefault(c, []).append(z)
    assert all(len(v) >= 2 for v in inv.values())


def test_parse_label_invalid_counted():
    assert agents.parse_label("3", K) == 2
    assert agents.parse_label("Answer: 8.", K) == 7
    assert agents.parse_label("9", K) == -1 and agents.parse_label("none", K) == -1 and agents.parse_label("", K) == -1
    b, c, R = rewards_from([0, 1, -1], [-1, 1, 2], [0, 1, 2], 1.0)
    assert list(c) == [0, 1, 1] and list(b) == [1, 1, 0] and list(R) == [1.0, 0.0, -1.0]


def test_clean_message():
    assert agents.clean_message("", 'A: "Hello there friend. And then the') == "Hello there friend."
    assert agents.clean_message("Anyway, x.", "Nice weather.").startswith("Anyway, x. Nice")


def test_device_auto_resolution():
    dev = resolve_device("auto")
    assert dev in ("cuda", "mps", "cpu")
    dtype = resolve_dtype(dev)
    assert dtype in ("float16", "float32")
    assert resolve_dtype("cpu", "float16") == "float32"


def test_reinforce_loss_sign_and_detach():
    pol = CoverController(4, 2, 3, 4, 8, init_std=0.5)
    f, z = torch.randn(5, 4), torch.zeros(5, dtype=torch.long)
    d = pol.dist(f, z)
    a = torch.tensor([1, 1, 1, 1, 1])
    r = torch.ones(5, requires_grad=True)
    loss, logp, adv, _ = reinforce_loss(d, a, r, 0.25, 0.0)
    assert not adv.requires_grad and logp.requires_grad


def test_tracker_offline_safety():
    cfg_copy = copy.deepcopy(CFG)
    cfg_copy["wandb"] = {"enabled": False}
    tracker = ExperimentTracker(cfg_copy, job_type="unit-test")
    # Tracker should log without crashing even when disabled
    tracker.log({"dummy_metric": 1.0}, step=1)
    tracker.finish()


def test_linguistic_audit_metrics():
    from proto_1.linguistic_audit import distinct_ngrams, repetition_ratio, tokenize_words
    words = tokenize_words("Hello, world! This is a test.")
    assert words == ["hello", "world", "this", "is", "a", "test"]
    d1 = distinct_ngrams(["hello world", "hello friend"], n=1)
    assert 0.0 < d1 <= 1.0
    rep_positive = repetition_ratio(["the the test", "normal sentence"])
    assert rep_positive == 0.5
    rep_zero = repetition_ratio(["one two three", "four five six"])
    assert rep_zero == 0.0

