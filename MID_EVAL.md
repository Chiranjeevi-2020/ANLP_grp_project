# Mid-evaluation: Learned Controllers for Latent-Preserving Context Switching

**Status:** the full experiment (`./run_all.sh`, `configs/full.json`) completed in 2 h 03 min on an Apple M4, and all 48 audit checks passed. The numbers below are copied from `results/mideval/full/eval/RESULTS.md`.

## What was built (scope from the proposal, Weeks 1–6 subset)

- **Task.** A and B share a hidden topic z. Outsider C joins, and A must switch the surface topic so that insider B can still recover z while C cannot.
  - There are 8 benign fictional topics in 4 pairs, for example a surprise party vs a farewell video, or a club trivia night vs a new café venue.
  - There are 8 shared cover topics.
  - The scripted cover map is two-to-one, so a cover is not a label code.
- **Dataset.** Templated pre-switch histories are written at the pair level, with an identical string for both topics in a pair. So history alone caps B at 50 %, and the post-switch message has to do the rest.
  - Splits are by template family: 6 train, 2 dev and 2 test families per pair.
  - Duplicates and near-duplicates are checked across splits.
  - Labels live only in separate metadata files.
- **Frozen model:** `Qwen2.5-1.5B-Instruct`, fp16, on Apple MPS. It plays A (generator), prompted B, prompted C, and serves as the feature extractor. No base weights are updated.
- **Baselines:** no-switch (reference), scripted switch (fixed map plus transition template; A is not told z), and prompt-only switch (A picks a cover itself). All share the same decoding settings and token budget.
- **B decoder (reward):** a linear probe on frozen mean-pooled features of B's permitted text (history and message).
  - Trained on train-split baseline rollouts only. Layer and weight decay are selected on dev, and the probe is frozen before RL.
  - A reward gate checks that it beats the history-only decoder on dev.
- **C:** a fixed prompted classifier. It sees only the post-switch public text. Invalid answers count as wrong.
- **Controller:** a categorical policy over 8 cover topics (about 0.5 M parameters), taking z's embedding and frozen history features as input. Its action becomes an explicit instruction in A's prompt.
- **REINFORCE at λ = 1:**
  - Reward: R = 1[B] − 1[C].
  - Loss: negative (detached advantage × log-probability), plus an entropy bonus.
  - The advantage uses the previous moving-average baseline.
  - Gradient clipping, a fixed seed, and checkpoint selection on dev expected reward.
  - It runs on a cached bank of train rollouts, with a live-LM smoke test first.
- **Evaluation on the untouched test split:** fresh rollouts for every method.
  - Metrics: B and C accuracy, joint success, mean reward with a cluster-bootstrap CI, valid rates, per-topic numbers, and Wilson CIs. Prompted B is reported alongside the decoder.
- **Controls:**
  1. B history-only (decoder and prompted).
  2. B message-only (decoder and prompted).
  3. Shuffled post-switch messages.
  4. A public-transcript-trained decoder, which is the matched-information audit. In this design it is the same model as the message-only decoder, because C's view is exactly the message.
- **Three audits** (data, smoke, final) generate `AUDIT.md`. There are also 11 unit tests (`tests/`).

## Results (run `full`, run_id `2123323ddf4f`; copied from `results/mideval/full/eval/RESULTS.md`)

Test split: 80 scenarios x 2 rollouts = 160 per method. Chance is 0.125; the history-only ceiling is 0.5. PRELIMINARY: one seed, one lambda.

| Method | B decoder | C prompted | Joint | Mean reward [95% CI] | B prompted |
|---|---|---|---|---|---|
| no_switch | 0.875 | 0.637 | 0.269 | 0.237 [0.15,0.33] | 0.644 |
| scripted | 0.519 | 0.181 | 0.406 | 0.338 [0.23,0.44] | 0.306 |
| prompt_only | 0.656 | 0.431 | 0.294 | 0.225 [0.13,0.32] | 0.519 |
| untrained_controller | 0.425 | 0.156 | 0.319 | 0.269 [0.17,0.36] | 0.331 |
| rl_controller | 0.637 | 0.144 | 0.500 | 0.494 [0.41,0.58] | 0.269 |
| rl_controller_greedy | 0.606 | 0.100 | 0.506 | 0.506 [0.42,0.59] | 0.287 |

**Controls:**
- B history-only decoder: 0.450.
- RL controller with shuffled messages: 0.450.
- Public-transcript-trained decoder on RL messages: 0.300 (prompt-only 0.494, no-switch 0.781).

**RL:** dev expected reward rose from 0.218 at initialization to 0.443 at step 300 (the selected checkpoint). Entropy collapsed to a near-deterministic, history-keyed cover choice.

**Audits:** 13/13 data, 19/19 smoke, 16/16 final.

**Main caveats:**
- The gain does not transfer to prompted B.
- Prompted C is biased toward label 3.
- A trained outsider still recovers z at 2.4x chance.

The full analysis is in `report/main.pdf`.

## How to read the results (pre-registered interpretation rules)

- The RL controller "helps" only if its test reward and joint success exceed **both** prompt-only and the untrained controller, with non-overlapping (or nearly non-overlapping) intervals. Otherwise, report it as inconclusive.
- B's accuracy only shows post-switch information transfer if it exceeds the **history-only** control (around 0.5 at most) and the **shuffled-message** control.
- If the public-transcript decoder recovers z well while prompted C does not, the apparent "hiding" comes from C being weak, not from secure communication.
- A single λ means no trade-off curve can be claimed. A small test set means no significance claims.

## Limitations (also in `AUDIT.md`)

- B (a trained probe) and C (prompted) are asymmetric. Prompted-B vs prompted-C is the like-for-like comparison.
- C is a small, non-adaptive eavesdropper, and all agents share one base model.
- The histories are templated, and the topic set is small and fixed.
- The cached rollout bank (k = 2 per state-action) can be exploited by the policy. Fresh test rollouts measure this.
- One seed, one λ, and roughly 160 test rollouts per method (80 examples × 2).

## Post-mid-evaluation plan

1. Run λ ∈ {0.5, 1, 2} and multiple seeds for the RQ3 trade-off curve.
2. Make C stronger: a larger prompted model, and a C decoder trained on public transcripts in the reward (adaptive adversary).
3. Compare soft-prompt conditioning with discrete conditioning (Lester et al., 2021).
4. Generate B's reply and do multi-turn rollouts.
5. Run the RQ4 linear probe on controller representations.
6. Add LLM-generated rather than templated histories, more topics, and held-out cover topics.
7. Do a light naturalness check of switched turns.
