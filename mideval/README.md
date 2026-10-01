# `mideval/` — mid-evaluation prototype

Learned discrete controller for latent-preserving context switching, trained with REINFORCE at λ = 1,
on top of a frozen `Qwen/Qwen2.5-1.5B-Instruct` (fp16, Apple MPS). Self-contained. The older `src/` code
(soft prefixes, leakage gate, probes) is not used for this milestone.

## Commands

```bash
pip install "transformers>=4.45,<5" accelerate pytest torch numpy matplotlib
./run_all.sh                         # full run (configs/full.json), ~2-2.5 h on an M4 / 16 GB
./run_all.sh configs/tiny.json       # ~20 min pipeline check; its numbers are NOT results
# or stage by stage:
python3 -m mideval.cli build-data       --config configs/full.json --overwrite   # archives any old run
python3 -m mideval.cli audit-data       --config configs/full.json               # AUDIT 1
python3 -m mideval.cli gen-decoder-data --config configs/full.json               # baseline rollouts on train/dev
python3 -m mideval.cli train-decoder    --config configs/full.json               # B decoder + control decoders
python3 -m mideval.cli build-bank       --config configs/full.json               # train/dev rollouts for every action
python3 -m mideval.cli smoke            --config configs/full.json               # live-LM REINFORCE smoke test
python3 -m mideval.cli audit-smoke      --config configs/full.json               # AUDIT 2
python3 -m mideval.cli train-rl         --config configs/full.json               # REINFORCE, dev checkpoint selection
python3 -m mideval.cli evaluate         --config configs/full.json               # fresh TEST rollouts, all methods
python3 -m mideval.cli audit-final      --config configs/full.json               # AUDIT 3
python3 -m pytest -q tests                                                       # unit tests (no LM)
```

## Task and information boundaries

| Party | Receives | Never receives |
|---|---|---|
| A (frozen LM) | history, private note with z's description, cover instruction (method-dependent) | — |
| Controller | z (embedding) + frozen LM features of B-visible history | test data |
| B (insider) | pre-switch history + `[C joins the conversation]` + A's first post-switch message | z, its description, action, controller state, A's prompt |
| C (outsider) | `[C joins the conversation]` + A's first post-switch message | all of the above **and** the history |

* B and C choose from the same complete, fixed-order list of 8 labels.
* **Primary measurement** is decoding after A's first post-switch utterance. No B reply is generated, so B's reply cannot create the signal.
* Histories are written at the **topic-pair** level. Both topics of a pair get the *identical* history string, so history alone gives at most 50 %.
* **Asymmetries (documented, intentional):**
  * B's *reward* decoder is a trained linear probe on frozen features, while C is a fixed prompted classifier.
  * The matched-information audit trains the *same* decoder architecture on C's public view only (`message` decoder). In this design, C's public view and "B message-only" are the same text, so controls 2 and 4 use one decoder.
  * Prompted B is also reported as a like-for-like (untrained) comparison with prompted C.

## Methods (matched decoding: max_new_tokens=60, T=0.8, top-p=0.95, same test cases/labels)

| name | A told z? | cover topic chosen by |
|---|---|---|
| `no_switch` | yes | none: A continues openly (reference, not a guaranteed ceiling) |
| `scripted` | no | fixed map `scripted_map` (each used cover ↔ 2 hidden topics) + fixed transition template |
| `prompt_only` | yes | A itself, from the cover list |
| `untrained_controller` | yes | controller at initialization (near-uniform), sampled |
| `rl_controller` | yes | REINFORCE controller (dev-selected checkpoint), sampled |
| `rl_controller_greedy` | yes | same, argmax action |

## RL details

* Policy: `CoverController`. It takes z → 64-d embedding, concatenated with the standardized history feature (1536-d, the layer chosen for the B decoder), then a 2×256 MLP that outputs 8 cover-topic logits. The head is initialized near zero, so the initial policy is close to uniform. About 0.5 M parameters.
* Reward: `R = 1[B_decoder(history, msg) = z] − λ·1[C_prompted(msg) = z]`, with λ = 1, so R ∈ {−1, 0, 1}.
* Loss: `−mean((R − b_prev).detach() · log π(a|s)) − β·H(π)`, with β = 0.01, gradient clipping at 1.0 and Adam at 3e-3.
  * `b_prev` is the moving-average baseline *before* this batch. It is updated afterwards with momentum 0.9.
* **Cached environment.** Before RL, A's message is generated `k` times for each (train or dev state, action). C and the frozen B decoder score each message once. Each RL step samples actions from the policy and then one of the k cached rollouts.
  * This is ordinary on-policy REINFORCE over a stochastic environment. No gradient passes through text or classifier outputs.
  * Test rollouts are generated fresh, for every method.
* Checkpoint selection uses the exact expected dev reward under the policy (Σ_a π(a|s)·mean_k R). The dev bank is never used for updates.
* The smoke test runs the same `reinforce_step` with the **live** LM in the loop for a few steps, and checks gradients, weights and reward values.

## Output layout (`results/mideval/<run_name>/`)

```
manifest.json                     run_id (stamped on every artifact), config
data/{split}_public.jsonl         {example_id, history:[[speaker, text], ...]}      <- B-visible
data/{split}_labels.jsonl         {example_id, z, pair, family, fill}               <- private eval metadata
data/data_report.json             counts, families, duplicate check
features/{split}_hist.npy         [n_examples, n_layers, 1536] fp16, aligned to *_public.jsonl
rollouts/{split}/<name>.public.jsonl   {rid, example_id, message}                   <- B/C-visible
rollouts/{split}/<name>.private.jsonl  {rid, example_id, method, action, rollout, a_system, a_user, raw}
rollouts/{split}/<name>.msgfeat.npy    message features aligned to the public file
rollouts/{train,dev}/bank.scores.jsonl {rid, example_id, action, rollout, b_pred, c_pred, c_raw, b_correct, c_correct, reward}
decoders/{full,history,message}.pt + report.json (grid, dev accuracy per method, reward gate)
rl/smoke.json, controller_init.pt, controller_best.pt, curve.json, report.json, training_curve.png
eval/metrics.json, RESULTS.md, per_topic.csv, examples.md, raw_<method>.jsonl (per-rollout predictions + gold)
audit/{data,smoke,final}.json, AUDIT.md, timings.json
```
`rid` and `example_id` are opaque hashes. Invalid prompted answers get `pred = -1` and count as incorrect.

Ethics (per proposal): controller checkpoints are git-ignored (`results/`), so they are not released.
