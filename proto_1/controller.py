"""Discrete cover-topic controller and the REINFORCE update for proto_1.

Maps z and frozen history features to a categorical policy over cover topics.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical


class CoverController(nn.Module):
    def __init__(self, feat_dim, n_topics, n_actions, topic_emb_dim=64, hidden_dim=256, init_std=0.01,
                 feat_mean=None, feat_std=None):
        super().__init__()
        self.register_buffer("feat_mean", torch.zeros(feat_dim) if feat_mean is None else torch.as_tensor(feat_mean))
        self.register_buffer("feat_std", torch.ones(feat_dim) if feat_std is None else torch.as_tensor(feat_std))
        self.topic_emb = nn.Embedding(n_topics, topic_emb_dim)
        self.net = nn.Sequential(nn.Linear(feat_dim + topic_emb_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                 nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.head = nn.Linear(hidden_dim, n_actions)
        nn.init.normal_(self.head.weight, std=init_std)  # near-uniform initial policy
        nn.init.zeros_(self.head.bias)

    def forward(self, feats, z):
        x = (feats - self.feat_mean) / self.feat_std
        return self.head(self.net(torch.cat([x, self.topic_emb(z)], -1)))

    def dist(self, feats, z):
        return Categorical(logits=self(feats, z))


class MovingBaseline:
    """Advantage uses the PREVIOUS baseline value; the baseline is updated afterwards."""

    def __init__(self, momentum=0.9, init=0.0):
        self.momentum, self.value = momentum, float(init)

    def advantage(self, rewards):
        return (rewards - self.value).detach()

    def update(self, rewards):
        self.value = self.momentum * self.value + (1 - self.momentum) * float(rewards.mean())


def reinforce_loss(dist, actions, rewards, baseline_value, entropy_coef):
    logp = dist.log_prob(actions)                        # keeps the graph to the controller
    adv = (rewards.float() - baseline_value).detach()    # rewards/baseline carry no gradient
    entropy = dist.entropy().mean()
    loss = -(adv * logp).mean() - entropy_coef * entropy
    return loss, logp, adv, entropy


def reinforce_step(policy, opt, feats, z, reward_fn, baseline, rl_cfg, generator):
    """One on-policy update. reward_fn(actions LongTensor) -> rewards tensor in {-1,0,1}."""
    policy.train()
    d = policy.dist(feats, z)
    probs = d.probs.detach()
    assert torch.isfinite(probs).all() and torch.allclose(probs.sum(-1), torch.ones(len(probs)), atol=1e-4)
    actions = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    rewards = torch.as_tensor(reward_fn(actions), dtype=torch.float32).detach()
    loss, logp, adv, ent = reinforce_loss(d, actions, rewards, baseline.value, rl_cfg["entropy_coef"])
    opt.zero_grad()
    loss.backward()
    gnorm = float(nn.utils.clip_grad_norm_(policy.parameters(), rl_cfg["grad_clip"]))
    opt.step()
    baseline.update(rewards)
    return {
        "loss": float(loss.item()), "grad_norm": gnorm, "grad_finite": bool(np.isfinite(gnorm)),
        "entropy": float(ent.item()), "reward_mean": float(rewards.mean().item()),
        "adv_abs_mean": float(adv.abs().mean().item()), "baseline_before": float(baseline.value),
        "actions": actions.tolist(), "rewards": rewards.tolist(),
    }


import numpy as np  # used above for isfinite
