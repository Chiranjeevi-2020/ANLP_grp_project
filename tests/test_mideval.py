"""Fast unit tests (no LM needed). Run: python -m pytest -q tests"""
import copy

import numpy as np
import pytest
import torch

from mideval import agents, decoders
from mideval.config import load_config, load_spec
from mideval.controller import CoverController, MovingBaseline, reinforce_loss, reinforce_step
from mideval.data import build_examples, duplicate_report
from mideval.evaluate import cluster_bootstrap, wilson
from mideval.pipeline import rewards_from

SPEC = load_spec()
CFG = load_config("configs/full.json")
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
    assert all(utt not in cu for _, utt in r["history"])          # C joins after the switch
    assert all(utt in bu for _, utt in r["history"])              # B is an insider
    assert agents.label_list(SPEC) in cu and agents.label_list(SPEC) in bu  # same full label list
    _, au = agents.a_prompt(SPEC, "controller", r["history"], p, action=3, example_id=r["example_id"])[:2]
    assert secret in au and SPEC["cover_topics"][3]["label"] in au   # only A gets z and the action
    _, su, pre = agents.a_prompt(SPEC, "scripted", r["history"], p, example_id=r["example_id"])
    assert secret not in su and pre in su                              # scripted A is not told z


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
    assert list(c) == [0, 1, 1] and list(b) == [1, 1, 0] and list(R) == [1.0, 0.0, -1.0]  # invalid C != correct


def test_clean_message():
    assert agents.clean_message("", 'A: "Hello there friend. And then the') == "Hello there friend."
    assert agents.clean_message("Anyway, x.", "Nice weather.").startswith("Anyway, x. Nice")


def test_reinforce_loss_sign_and_detach():
    pol = CoverController(4, 2, 3, 4, 8, init_std=0.5)
    f, z = torch.randn(5, 4), torch.zeros(5, dtype=torch.long)
    d = pol.dist(f, z)
    a = torch.tensor([1, 1, 1, 1, 1])
    r = torch.ones(5, requires_grad=True)
    loss, logp, adv, _ = reinforce_loss(d, a, r, 0.25, 0.0)
    assert not adv.requires_grad and logp.requires_grad
    assert torch.allclose(loss, -(0.75 * logp).mean())
    loss.backward()
    assert r.grad is None  # no gradient flows into rewards


def test_reinforce_step_learns_and_respects_baseline_order():
    torch.manual_seed(0)
    pol = CoverController(4, 2, 3, 4, 8, init_std=0.5)
    opt = torch.optim.Adam(pol.parameters(), lr=0.05)
    f, z = torch.randn(32, 4), torch.zeros(32, dtype=torch.long)
    g = torch.Generator().manual_seed(0)
    base = MovingBaseline(0.9)
    w0 = copy.deepcopy(pol.state_dict())
    with torch.no_grad():
        p0 = pol.dist(f, z).probs[:, 2].mean().item()
    prev = base.value
    for _ in range(60):
        st = reinforce_step(pol, opt, f, z, lambda a: (a == 2).float() - (a == 0).float(), base,
                            {"entropy_coef": 0.0, "grad_clip": 1.0}, g)
        assert st["baseline_before"] == prev and st["grad_finite"] and set(st["rewards"]) <= {-1, 0, 1}
        prev = st["baseline_after"]
    with torch.no_grad():
        p1 = pol.dist(f, z).probs[:, 2].mean().item()
    assert p1 > p0 + 0.2
    assert any(not torch.equal(w0[k], v) for k, v in pol.state_dict().items())


def test_decoder_frozen_after_fit():
    X = np.random.randn(40, 6).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    m = decoders.fit(X, y, 2, 0.01, 100, 0.05, 0)
    assert not any(p.requires_grad for p in m.parameters()) and (m.predict(X) == y).mean() > 0.8
    ck = m.checksum()
    m.predict(X)
    assert m.checksum() == ck


def test_intervals():
    lo, hi = wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
    lo, hi = cluster_bootstrap([1, 0, 1, 1], ["a", "a", "b", "b"], 200, 0)
    assert lo <= 0.75 <= hi
