# Proto 1: Learned Controllers for Latent-Preserving Context Switching in Multi-Agent LM Dialogue

This folder contains the complete, upgraded **Proto 1** implementation for the ANLP project.

---

## 🌟 What's New in Proto 1

1. **Hardware Auto-Detection:**
   - Robust device priority: `CUDA` > `MPS` > `CPU`.
   - Automated half-precision (`float16`) on CUDA / Apple Silicon MPS, and `float32` on CPU.
2. **Weights & Biases (WandB) Integration:**
   - Full live tracking of RL training (loss, advantage, reward, baseline, entropy, gradient norm).
   - Evaluation tracking across all baselines and controls (mean reward, B/C accuracy, joint success, keyword leakage).
   - Graceful offline / local fallback if no API key is provided.
3. **Hugging Face Hub Checkpointing:**
   - Automatic checkpoint and results uploading to Hugging Face model repository when `HF_TOKEN` is supplied.
4. **Multi-Lambda Sweep (`sweep_lambda.py`):**
   - Empirical Pareto frontier computation across $\lambda \in \{0.0, 0.5, 1.0, 2.0\}$.
   - Plots insider accuracy vs. outsider obfuscation trade-offs.
5. **Linguistic Naturalness & Lexical Diversity Audit (`linguistic_audit.py`):**
   - Evaluates Perplexity (PPL) of post-switch messages under the frozen base LM.
   - Computes Distinct-1, Distinct-2, word lengths, and repetition rates to prove naturalness without language drift.
6. **Theoretical Grounding & Literature Updates:**
   - Formal connection to the Rational Speech Acts (RSA) overhearer-aware speaker formulation ($R = \mathbb{1}[B=z] - \lambda \mathbb{1}[C=z]$).
   - Integration of Representation Engineering (RepE, ITI) and Multi-Party Dialogue Floor Management literature in `proto_1/report/`.

---

## 📁 Directory Structure

```
proto_1/
├── __init__.py               # Package metadata
├── config.py                 # Configuration loader and file I/O
├── lm.py                     # FrozenLM with CUDA/MPS/CPU auto-detection & perplexity calculation
├── agents.py                 # Prompt builders and boundary enforcement for A, B, C
├── controller.py             # Discrete CoverController policy & REINFORCE update
├── decoders.py               # Supervised linear probes on frozen features
├── data.py                   # Pair-ambiguous synthetic dataset generator
├── rollouts.py               # Message generation & feature extraction
├── pipeline.py               # Full pipeline stages with WandB tracking
├── evaluate.py               # Test split evaluation, bootstrap CIs, WandB logging
├── audit.py                  # Data, smoke, and final leakage verification audits
├── tracker.py                # WandB & Hugging Face Hub tracking utilities
├── cli.py                    # Command-line entry point
├── sweep_lambda.py           # Multi-lambda sweep for RQ3 Pareto trade-off curve
├── linguistic_audit.py       # Perplexity and lexical diversity audit
├── run_colab.py              # Standalone 1-click execution script for Google Colab
├── configs/
│   ├── colab_fast.json       # Fast Colab preset (~10-15 min on T4 GPU)
│   ├── full.json             # Full scientific paper preset (~2 hours)
│   └── tiny.json             # Fast sanity check (~1 min)
├── data/
│   └── spec.json             # 8 hidden topics, 8 cover topics, pair templates
├── tests/
│   └── test_proto_1.py       # Comprehensive unit tests
└── report/
    ├── main.tex              # Updated LaTeX paper with RSA grounding
    ├── refs.bib              # BibTeX with RSA, RepE, and multi-party literature
    ├── main.pdf              # Compiled 10-page paper
    └── figures/              # Experimental diagrams and result plots
```

---

## 🚀 Quickstart

### Option A: Run in Google Colab (Recommended for Free GPU)
1. Open [`notebooks/run_proto_1_colab.ipynb`](../notebooks/run_proto_1_colab.ipynb) in Google Colab.
2. Under **Runtime → Change runtime type**, select **T4 GPU**.
3. Run the cells step-by-step!

### Option B: Run via CLI Locally or on Cloud VM
```bash
# 1. Run the fast Colab configuration
python -m proto_1.run_colab --config proto_1/configs/colab_fast.json

# 2. Run the multi-lambda sweep (RQ3 Pareto frontier)
python -m proto_1.sweep_lambda --config proto_1/configs/colab_fast.json --lambdas 0.0 0.5 1.0 2.0

# 3. Run the linguistic naturalness and perplexity audit
python -m proto_1.linguistic_audit --config proto_1/configs/colab_fast.json

# 4. Run fast unit tests
python -m pytest -q proto_1/tests
```

---

## 🔑 Setting up WandB and Hugging Face Hub (Optional)

You can either pass keys interactively when prompted by `run_colab.py`, or export them in your shell:
```bash
export WANDB_API_KEY="your_wandb_api_key"
export WANDB_PROJECT="anlp-context-switching"

export HF_TOKEN="your_huggingface_write_token"
export HF_REPO_ID="username/anlp-context-switching-controller"
```
If skipped, all logs and checkpoints are saved locally under `results/proto_1/`.
