# 🚀 Google Colab T4 GPU Execution & Headless Secrets Sync Guide

This guide provides foolproof, step-by-step instructions for running our Week 8 comprehensive pipeline with **100% headless, non-interactive authentication** to **Weights & Biases (W&B)**, **Hugging Face Hub (HF)**, and **GitHub**.

---

## 1. Quick Decision Matrix: Which Notebook Should You Run?

| Notebook | Purpose | Estimated T4 GPU Runtime | When to Use |
| :--- | :--- | :---: | :--- |
| **`01_Week8_Comprehensive_Pipeline.ipynb`** | End-to-end benchmark: Multi-$\\lambda$ sweep, Soft-Prefix vs. Discrete showdown, Layer-wise interpretability probing, Informed-B evaluation, Linguistic audit, and automated sync to W&B, HF, and GitHub. | **~35 – 40 mins** (Full) / **~10 – 12 mins** (Fast) | Run this tonight to generate, sync, and verify all experimental figures, models, and metrics. |
| **`02_Interactive_Demonstrator.ipynb`** | Real-time live demo: Input any topic $z$, watch Agent A switch topics, and see B and C classify live with probability bars. | **Instant (< 10 seconds per turn)** | Use this during tomorrow's 30-minute viva when the TA says: *"Show me a live demonstration."* |

---

## 2. Setting Up the Colab Secrets Vault (Zero Interactive Prompts)

To achieve **100% autonomous headless execution without pausing for manual prompts or logins**, all credentials are read directly from the Google Colab Secrets Vault:

### Step 1: Open the Secrets Vault in Colab
1. Open the notebook in Google Colab ([colab.research.google.com](https://colab.research.google.com)).
2. Look at the left sidebar of Google Colab and click the **Key icon 🔑** ("Secrets").

### Step 2: Add the 3 Secrets
Click **"+ Add new secret"** for each of the following:

| Secret Name | Where to Get It | Purpose |
| :--- | :--- | :--- |
| **`WANDB_API_KEY`** | [wandb.ai/authorize](https://wandb.ai/authorize) | Streams real-time RL loss curves, dev accuracy, and Pareto frontier plots to your W&B dashboard. |
| **`HF_TOKEN`** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) *(Write permission)* | Uploads trained controller checkpoints and model cards to Hugging Face Hub. |
| **`GITHUB_TOKEN`** | [github.com/settings/tokens](https://github.com/settings/tokens) *(repo scope)* | Headlessly clones private/public repo, commits new results, and pushes directly to `origin/main`. |

### Step 3: Toggle "Notebook access" ON
> ⚠️ **CRITICAL:** For each secret, make sure the toggle switch **"Notebook access"** is turned **ON** (blue). If disabled, Colab will block the notebook from reading the keys.

*(Note: If you leave any secret blank, the notebook will automatically and safely fall back to local offline logging without crashing).*

---

## 3. Activating the GPU Accelerator (Crucial Step!)

Google Colab defaults to CPU mode. You **must** switch to the free NVIDIA T4 GPU:
1. In the top navigation bar of Google Colab, click **Runtime** $\rightarrow$ **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU**.
3. Under **GPU type**, ensure **Standard** is selected.
4. Click **Save**.
5. When `Step 1` runs, it will verify:
   ```text
   PYTORCH & CUDA SYSTEM AUDIT (Target: 16GB T4 GPU)
   PyTorch Version : 2.x
   CUDA Available  : True
   GPU Device Name : Tesla T4
   Total VRAM      : 15.77 GB
   Memory Cache    : Cleared & Optimized for Float16 operations
   ```

---

## 4. One-Click Execution & Automated Sync Pipelines

Once your secrets are added and GPU is selected, click **Runtime** $\rightarrow$ **Run all** (or press `Ctrl + F9`). The notebook runs completely autonomously:

1. **GitHub Headless Sync:**
   - Authenticates using `https://{GITHUB_TOKEN}@github.com/Chiranjeevi-2020/ANLP_grp_project.git`.
   - Clones or pulls without any login dialogs.
2. **Weights & Biases Live Streaming:**
   - Initializes `wandb.init(project="ANLP-Latent-Context-Switching")`.
   - Logs moving-average return, advantage, policy entropy, dev reward, and Pareto trade-off points in real time.
3. **Hugging Face Hub Model Upload:**
   - Automatically creates repository `https://huggingface.co/<username>/anlp-latent-switch-controller`.
   - Uploads `controller_best.pt`, `controller_init.pt`, and `curve.json`.
4. **Automated GitHub Results Push:**
   - Automatically commits newly generated results, tables, and Pareto plots back to the repository on GitHub.

---

## 5. End-to-End Live Verification & Defense Links

The final cell in the notebook prints live clickable URLs so you can show the TA live cloud verification during the viva:
- 📊 **Weights & Biases Dashboard:** `https://wandb.ai/<user>/ANLP-Latent-Context-Switching/runs/<run_id>`
- 🤗 **Hugging Face Model Card:** `https://huggingface.co/<user>/anlp-latent-switch-controller`
- 🐙 **GitHub Latest Commit:** `https://github.com/Chiranjeevi-2020/ANLP_grp_project/commit/<sha>`

---

## 6. Exact Execution Breakdown & T4 GPU ETAs (Plan Your Time Tonight)

```
+-------------------------------------------------------------+-------------------+
| Pipeline Stage                                              | Execution Time    |
+-------------------------------------------------------------+-------------------+
| 1. Headless Secrets Retrieval & CUDA VRAM Audit             | ~5 sec            |
| 2. Environment installation & Headless Git Sync            | ~1 min            |
| 3. Base Model Weight Loading (Qwen2.5-1.5B in float16)     | ~25 sec           |
| 4. Synthetic Paired-History Dataset Generation             | ~2 sec            |
| 5. Baseline Rollouts for B Decoder Pretraining             | ~6 - 8 mins       |
| 6. Linear Decoder Pretraining (Layer 28 Residual State)     | ~20 sec           |
| 7. Cached Rollout Bank Creation (3,072 Train + 512 Dev)     | ~18 - 22 mins     |
| 8. REINFORCE Live Smoke Test (3 gradient steps)             | ~30 sec           |
| 9. REINFORCE RL Controller Training (1,000 steps)           | ~6 sec            |
| 10. Deliverable 1: Multi-Lambda Sweep (RQ3 Pareto Curve)    | ~3 - 4 mins       |
| 11. Deliverable 2: Soft-Prefix vs. Discrete Showdown       | ~2 mins           |
| 12. Deliverable 3: Layer-wise Interpretability Probing (RQ4)| ~1 min            |
| 13. Deliverable 4: Pragmatic Informed-B Evaluation         | ~1 min            |
| 14. Deliverable 5: Linguistic Perplexity & Fluency Audit   | ~1.5 mins         |
| 15. Automated Sync: Hugging Face & GitHub Commit           | ~45 sec           |
+-------------------------------------------------------------+-------------------+
| TOTAL RUNTIME (Full Benchmark Suite):                       | ~36 - 42 minutes  |
+-------------------------------------------------------------+-------------------+
| FAST PRESET (proto_1/configs/colab_fast.json):              | ~10 - 12 minutes  |
+-------------------------------------------------------------+-------------------+
```
