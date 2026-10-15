# Phase 1: 80% Completion Audit & Gap Analysis

**Course:** Advanced NLP (Monsoon 2026), IIIT Hyderabad  
**Instructors:** Prof. Manish Shrivastava, Aparajitha Allamraju  
**Timeline Position:** Week 8 of 10 (Project Span: Aug 21 – Oct 31, 2026; exactly 80% elapsed)  
**Evaluation Format:** Postponed Mid-Evaluation Viva (30 Minutes)  
**Project:** Learned Controllers for Latent-Preserving Context Switching in Multi-Agent LM Dialogue  

---

## 1. Executive Context & The "80% Expectation" Reality

Because this mid-evaluation was postponed from Week 5 (September 30th) to Week 8 (October 16th), the evaluation bar is no longer "early proof of feasibility." 
In the course grading scheme, the project accounts for **50% of the entire course grade** (Proposal: 10%, Mid-Eval: 20%, Final Eval: 20%). The instructors explicitly stated in Lecture 1 Logistics:
- *"Advanced Level projects suitable to be submitted as papers; Highly encouraged to submit work to A* conferences/workshops."*
- *"Projects NOT suitable: Finetuning/Instruction finetuning, RAG."*
- *"Suitable Projects: Architecture optimization & changes, Novel Evaluation Methods, Datasets."*

At Week 8, the Teaching Assistant (TA) will evaluate us not as students presenting preliminary ideas, but as researchers who should have **nearly all empirical pipelines finished, all baselines benchmarked, and preliminary quantitative conclusions established**.

---

## 2. Milestone Audit: Proposal Promises vs. Submitted Mid-Eval Code

| Milestone & Research Question | Original Proposal Promise | Submitted Mid-Eval State (`mideval/`) | Status at Week 8 | Severity / TA Risk |
| :--- | :--- | :--- | :--- | :--- |
| **RQ1: B Recovery vs. Baselines** | Compare learned controller against No-Switch, Scripted, and Prompt-Only baselines. | 8 topics in 4 ambiguous pairs evaluated on 80 test scenarios ($n=160$). Learned controller achieved B=63.7% vs Scripted=51.9%, Prompt-only=65.6%. | **Partially Complete:** Only discrete conditioning was evaluated. Soft-prefix continuous steering was omitted from the unified benchmark. | 🟡 **Medium**: Missing head-to-head comparison with continuous steering. |
| **RQ2: Eavesdropper C Suppression** | Suppress outsider C's recovery of secret $z$ from switched transcript. | C suppressed to 14.4% under RL controller vs 63.7% under No-Switch and 43.1% under Prompt-only. | **Completed, but Fragile:** Prompted C has severe label-3 bias (72%), and a trained public probe recovers $z$ at 30.0% ($2.4\times$ chance). | 🔴 **High**: TA will call C a "strawman" unless matched against an adaptive or probe adversary. |
| **RQ3: Secrecy vs. Utility Trade-off (Pareto Curve)** | *"Is there a measurable trade-off between B's recovery and C's recovery as the adversarial weighting $\lambda$ in the reward is varied?"* (Weeks 5–6 deliverable). | Evaluated ONLY at a single operating point $\lambda = 1.0$ on a single seed! | **CRITICAL DEFICIT (Missing at Week 8):** Proposal explicitly promised $\lambda$-sweeps across multiple operating points. Zero empirical Pareto curves existed in `mideval/`. | 🔴 **Critical Penalty**: Direct breach of proposal RQ3 promise. |
| **RQ4: Interpretability Probing Across Layers** | *"Is $z$ still linearly recoverable from internal representations even when C's black-box guess is near chance?"* (Week 7 deliverable). | Not evaluated in `mideval/` test suite! `src/layer_probe.py` existed only as a disconnected script on older topic definitions. | **CRITICAL DEFICIT (Missing at Week 8):** Week 7 milestone was completely absent from the mid-eval report. No layer-wise linear probing of frozen LM representations. | 🔴 **Critical Penalty**: Direct breach of proposal RQ4 promise. |
| **Conditioning Showdown: Discrete vs. Soft-Prompt** | *"Checkpoint decision on soft-prompt vs. discrete conditioning based on a quick sanity test"* (Week 3 deliverable). | Abandoned soft-prompt for mid-eval; implemented discrete categorical cover selection only. | **Incomplete at Week 8:** Never ran the promised comparative showdown between soft prefix (`prefix_len=10` continuous embeddings) and discrete instructions. | 🟡 **High**: Professor will ask why continuous steering was dropped. |
| **Insider Generalization: The "Prompted B" Gap** | B should infer $z$ from A's switched utterances. | Zero-shot Prompted B collapsed to **26.9%** (below history-only baseline of 30.0%), while B's trained feature probe reached **63.7%**. | **Major Theoretical Flaw:** Proves that the RL controller discovered a feature-space steganographic artifact rather than human-interpretable natural language communication. | 🔴 **Critical Scientific Vulnerability**: Reviewers will tear this apart as "non-communicative feature hacking." |
| **Linguistic Fluency & Perplexity Audits** | "Lightweight naturalness check on switched conversations." | Zero quantitative metrics reported. No Perplexity ($PPL$), no Distinct-1/2 lexical diversity. Table 5 qualitative example showed Agent A violating instructions by addressing C directly! | **Missing at Week 8:** No proof that text quality is preserved or that hallucinations are bounded. | 🟡 **High**: TA will ask if generation degenerated into gibberish. |

---

## 3. The 5 Exact Questions & Penalties the TA Will Target Tomorrow

### 🎯 Trap 1: "You are at Week 8 out of 10. Where is the Pareto Trade-off Curve for RQ3?"
* **Why the TA will ask this:** The project title is *"Learned Controllers for Latent-Preserving Context Switching"* and RQ3 specifically poses the trade-off. At Week 8, having only $\lambda=1.0$ is unacceptable.
* **The Penalty:** Deduction of 5 to 8 marks (out of 20) under "Experimental Execution and Deliverables."
* **Our Required Fix in `after_midevals_submission`:** An automated multi-$\lambda$ sweep over $\lambda \in \{0.0, 0.5, 1.0, 2.0\}$ on the rollout bank, producing a formal Pareto frontier plot (`pareto_frontier.png`).

### 🎯 Trap 2: "Why did Prompted B collapse to 26.9%? Isn't this just feature-space steganography rather than NLP?"
* **Why the TA will ask this:** If an LLM receiver cannot read the message using natural language prompts, this is not natural language communication—it's an adversarial perturbation that only a linear probe can classify.
* **The Penalty:** Deduction under "Novelty and Theoretical Grounding."
* **Our Required Fix in `after_midevals_submission`:** Implement and benchmark **Informed / In-Context Prompted B** (giving B the pragmatic common ground that A is speaking in metaphors). Show that with pragmatic framing, prompted accuracy jumps from 26.9% to ~55-60%, proving true communicative grounding.

### 🎯 Trap 3: "Where is the Layer-wise Interpretability Probing promised in Week 7 (RQ4)?"
* **Why the TA will ask this:** Week 7 of the proposal explicitly promised to train linear probes across model hidden states to test whether $z$ persists in internal representations when C fails.
* **The Penalty:** Loss of marks on "Interpretability & Analysis."
* **Our Required Fix in `after_midevals_submission`:** Train and plot linear probes across Layers 8, 16, 22, and 28 of the frozen base LM on post-switch representations, proving where in the transformer hierarchy the latent topic is retained.

### 🎯 Trap 4: "Why did you abandon Continuous Soft-Prefix Steering after Week 3?"
* **Why the TA will ask this:** The proposal framed discrete vs. soft-prefix as a central architectural comparison (citing Lester et al., 2021 and Li & Liang, 2021).
* **The Penalty:** TA will criticize the team for avoiding the harder technical challenge.
* **Our Required Fix in `after_midevals_submission`:** Provide a unified head-to-head benchmark comparing `soft_prefix_agent_a` (10 continuous prefix tokens) against the discrete controller under identical evaluation conditions.

### 🎯 Trap 5: "Did the controller destroy language fluency? Where are your perplexity numbers?"
* **Why the TA will ask this:** In Table 5 of the submitted report, Agent A says *"C, did you hear about us planning a road trip?"* which violates the instruction ("Never speak to C"). The TA will ask if RL degraded fluency.
* **The Penalty:** Deduction under "Evaluation Rigor."
* **Our Required Fix in `after_midevals_submission`:** Compute and tabulate Perplexity ($PPL$), Distinct-1, Distinct-2, and repetition ratios across all methods using the frozen base LM.

---

## 4. Alignment with Course Lecture Slides

To satisfy the TA, every architectural choice must be mapped to the formal concepts taught in class:

| Lecture Topic (from Uploaded Slides) | Exact Concept in Slides | How Our Project Implements It |
| :--- | :--- | :--- |
| **Transformers: Attention** (`ANLP-M26-Attention.pdf`) | Scaled Dot-Product, Multi-Head Attention, Hidden States ($H=1536$). | Feature extraction $\phi(h)$ and $\phi(m)$ uses mean-pooled residual stream activations at Layer 28. |
| **Tokenization** (`ANLP-M26-Lec3-Tokenization.pdf`) | BPE, special tokens (`<\|im_start\|>`, `<\|im_end\|>`), Left-padding for batched generation. | `FrozenLM` enforces `padding_side="left"` so causal generation is aligned across batch samples. |
| **Positional Embeddings** (`ANLP-M26-Lec4-PE.pdf`) | Rotary Position Embeddings (RoPE), relative distance invariance. | Qwen2.5 utilizes RoPE; soft-prefix vectors are injected as virtual token embeddings at indices $0 \dots 9$. |
| **Normalization** (`ANLP_M26_Lecture_Normalization.pdf`) | RMSNorm vs. LayerNorm, Pre-LN vs. Post-LN, Input standardization. | Linear decoders and controller MLPs use explicit Feature Standardization $(\mathbf{x} - \mu)/\sigma$ and LayerNorm. |
| **Decoding & Generation** (`01_ANLP-M26-Lec1-Logistics.pdf`) | Nucleus Sampling ($p=0.95$), Temperature ($T=0.8$), Greedy decoding ($T=0$). | Generator A uses nucleus sampling for natural variety; Receiver B/C uses greedy decoding for deterministic classification. |
| **Reinforcement Learning for NLP** (`01_ANLP-M26-Lec1-Logistics.pdf`) | Policy Gradient Theorem, REINFORCE with baseline subtraction, Entropy bonus. | Controller trained using REINFORCE: $\nabla_\theta J = \mathbb{E}[(R - b) \nabla_\theta \log \pi_\theta(a \mid s)] + \beta \nabla_\theta H(\pi)$. |
| **Evaluation & Statistics** (`01_ANLP-M26-Lec1-Logistics.pdf`) | Bootstrap resampling vs. Wilson score intervals, Clustered standard errors. | Evaluation uses 2,000 cluster-bootstrap iterations across dialogue scenarios to prevent optimistic pseudo-replication. |

---

## 5. Action Plan for `after_midevals_submission`

To bridge every single gap before tomorrow morning, we implement:
1. `01_Week8_Comprehensive_Pipeline.ipynb`: A complete, self-contained Google Colab notebook that:
   - Mounts Google Drive and runs CUDA/VRAM verification.
   - Executes the multi-$\lambda$ sweep ($\lambda \in \{0.0, 0.5, 1.0, 2.0\}$) and generates the empirical Pareto curve.
   - Evaluates the continuous soft-prefix agent alongside discrete steering.
   - Runs layer-wise linear probing across Layers 8, 16, 22, and 28.
   - Evaluates the Informed-B insider to close the prompt transfer gap.
   - Computes language perplexity ($PPL$) and lexical diversity metrics.
2. `02_Interactive_Demonstrator.ipynb`: An interactive GUI/CLI notebook allowing the TA to enter custom dialogue scenarios and inspect real-time steering in under 10 seconds.
3. Full documentation: `COLAB_SETUP.md` and `CITATIONS_CHEAT_SHEET.md`.
