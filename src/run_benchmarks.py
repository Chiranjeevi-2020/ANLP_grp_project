import argparse
import csv
import json
import math
import random
import torch
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ProjectConfig, get_config
from src.base_model import FrozenBaseLM
from src.dialogue_manager import DialogueManager
from src.b_decoder import BDecoderHead
from src.soft_prefix_agent_a import SoftPrefixAgentA
from src.leakage_gate import LeakageGate

def compute_ppl(base_lm: FrozenBaseLM, text: str) -> float:
    """Computes perplexity = exp(cross_entropy_loss) on given text."""
    if not text.strip():
        return 1.0
    inputs = base_lm.tokenizer(text, return_tensors="pt").to(base_lm.device)
    with torch.no_grad():
        outputs = base_lm.model(inputs.input_ids, labels=inputs.input_ids)
        loss = outputs.loss.item()
    return math.exp(min(loss, 20.0))

class BenchmarkRunner:
    def __init__(self, config: Optional[ProjectConfig] = None):
        self.config = config or get_config()
        torch.set_num_threads(8)
        self.base_lm = FrozenBaseLM(self.config)
        self.dm = DialogueManager(self.base_lm, self.config)

        # Pretrained B Decoder
        self.b_decoder = BDecoderHead(
            base_hidden_dim=self.base_lm.hidden_dim,
            num_sensitive_topics=self.config.NUM_SENSITIVE_TOPICS
        ).to(self.base_lm.device)
        ckpt_path = self.config.CHECKPOINTS_DIR / "b_decoder_pretrained.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.base_lm.device)
            self.b_decoder.load_state_dict(ckpt["state_dict"])
            print(f"[Benchmark] Loaded BDecoderHead from {ckpt_path}")
        self.b_decoder.eval()

        # Soft Prefix Agent
        self.soft_agent = SoftPrefixAgentA(self.config)

        # Leakage Gate
        self.gate = LeakageGate(self.config)
        self._calibrate_gate()

        self.ppl_baseline = 15.0

    def _calibrate_gate(self):
        topic_0_kw = self.gate.get_keywords_for_topic(0)
        clean_calib = [
            "We are renovating the kitchen and picking new paint colors for the cabinets.",
            "The living room drywall looks great after the contractor finished patching it up.",
            "I'm considering wooden flooring instead of tiles for the master bedroom.",
            "Have you seen the new modern interior design catalogs for apartments?",
            "Replacing the old plumbing fixtures will save water in the long run.",
            "We picked neutral beige tones to make the small dining space feel bigger.",
            "The countertop installation is scheduled for this coming Saturday morning.",
            "Do you think recessed LED ceiling lights would look better in the hallway?",
            "We are comparing quotes from three different local remodeling contractors.",
            "Adding a fresh coat of primer helps the wall paint adhere much better.",
            "The tile backsplash in the kitchen really ties the entire design together.",
            "I am looking into energy-efficient double-pane glass windows for winter.",
            "The hardwood floor sanding created a lot of dust, but the finish is smooth.",
            "We are installing floating shelves along the study wall for book storage.",
            "A minimalist aesthetic really opens up small urban studio apartments.",
            "We need to buy more masking tape and drop cloths before painting the trim.",
            "The bathroom vanity came with brushed nickel handles that look very modern.",
            "Replacing the worn carpet with vinyl planks made a huge difference.",
            "We decided on quartz for the kitchen island because it is non-porous and durable.",
            "The architect suggested adding a skylight above the central staircase."
        ]
        self.gate.calibrate(clean_calib, topic_0_kw)

    def evaluate_turn(
        self,
        full_transcript: str,
        observed_by_c: str,
        topic_id: int
    ) -> Tuple[bool, bool]:
        """Runs Agent B (full history) and Agent C (post-entrance) evaluations."""
        # Agent B guess via decoder head (has full context from Turn 1)
        ctx_emb = self.base_lm.get_context_embedding(full_transcript)
        b_pred_id, _ = self.b_decoder.predict_topic(ctx_emb)
        b_correct = (b_pred_id == topic_id)

        # Agent C guess via zero-shot prompt (evaluates what C observed upon entrance)
        c_pred_id, _ = self.dm.query_agent_c_guess(observed_by_c)
        c_correct = (c_pred_id == topic_id)

        return b_correct, c_correct

    def run_all(self, runs_per_topic: int = 20) -> List[Dict[str, Any]]:
        methods = ["no_switch", "scripted", "soft_prefix"]
        all_records = []
        templates = self.dm.topics_data["naive_transition_templates"]
        mapping = self.dm.topics_data["naive_baseline_mapping"]

        total_runs = len(methods) * 10 * runs_per_topic
        print(f"\n=======================================================================")
        print(f"  Starting 3-Baseline Benchmark: {len(methods)} methods x 10 topics x {runs_per_topic} runs = {total_runs} runs")
        print(f"=======================================================================\n")

        pbar = tqdm(total=total_runs, desc="Benchmarking")
        run_counter = 0

        for method in methods:
            for topic_id in range(10):
                topic_info = self.dm.get_topic_by_id(topic_id)
                cover_id = int(mapping.get(str(topic_id), topic_id))
                cover_info = self.dm.get_cover_topic_by_id(cover_id)
                secret_keywords = topic_info["keywords"]

                for run_idx in range(runs_per_topic):
                    seed_val = 42 + run_counter
                    torch.manual_seed(seed_val)
                    random.seed(seed_val)
                    run_counter += 1

                    # Starting seed
                    seed_opener = topic_info["seeds"][run_idx % len(topic_info["seeds"])]
                    history = f"Agent A: {seed_opener}\nAgent B: I understand, let's talk through this carefully."

                    gate_blocked = False
                    reason = "safe"

                    if method == "no_switch":
                        # Agent A continues openly about topic z
                        open_continuation = f"Agent A: {seed_opener} We really need to resolve this {topic_info['name']} matter today."
                        full_transcript = f"{history}\n{open_continuation}"
                        utterance = open_continuation
                        observed_c = open_continuation
                        b_correct, c_correct = self.evaluate_turn(full_transcript, observed_c, topic_id)

                    elif method == "scripted":
                        # Scripted template switch
                        tpl = templates[run_idx % len(templates)]
                        scripted_turn = f"Agent A: " + tpl.format(cover_name=cover_info["name"])
                        b_followup = f"Agent B: Yes, {cover_info['name']} is definitely an interesting area."
                        full_transcript = f"{history}\n{scripted_turn}\n{b_followup}"
                        utterance = scripted_turn
                        observed_c = full_transcript
                        b_correct, c_correct = self.evaluate_turn(full_transcript, observed_c, topic_id)

                    elif method == "soft_prefix":
                        # Soft prefix generation
                        utt, _ = self.soft_agent.generate_with_soft_prefix(
                            topic_id=topic_id,
                            dialogue_history=history,
                            cover_topic_id=cover_id,
                            max_new_tokens=128
                        )
                        utterance = f"Agent A: {utt}"

                        # Gate check
                        is_safe, reason, beta = self.gate.check(utt, secret_keywords)
                        gate_blocked = not is_safe

                        # If gate blocks leakage, sanitize/filter so C is not tipped off
                        if gate_blocked:
                            filtered_utt = f"Agent A: Let's focus on our {cover_info['name']} plans for now."
                            full_transcript = f"{history}\n{filtered_utt}\nAgent B: Sounds good."
                            observed_c = f"{filtered_utt}\nAgent B: Sounds good."
                        else:
                            full_transcript = f"{history}\n{utterance}\nAgent B: That sounds like a solid plan."
                            observed_c = f"{utterance}\nAgent B: That sounds like a solid plan."

                        b_correct, c_correct = self.evaluate_turn(full_transcript, observed_c, topic_id)

                    ppl = compute_ppl(self.base_lm, utterance)
                    # Reward: int(b_correct) - 1.0*int(c_correct) - 0.5*max(0, ppl - ppl_baseline)
                    ppl_penalty = 0.5 * max(0.0, ppl - self.ppl_baseline)
                    reward = (1.0 if b_correct else 0.0) - (1.0 if c_correct else 0.0) - ppl_penalty

                    record = {
                        "method": method,
                        "topic_id": topic_id,
                        "topic_name": topic_info["name"],
                        "cover_topic_name": cover_info["name"],
                        "run_idx": run_idx,
                        "b_correct": b_correct,
                        "c_correct": c_correct,
                        "gate_blocked": gate_blocked,
                        "gate_reason": reason,
                        "ppl": ppl,
                        "reward": reward,
                        "utterance": utterance
                    }
                    all_records.append(record)
                    pbar.update(1)

        pbar.close()
        return all_records

def aggregate_and_save(records: List[Dict[str, Any]], out_csv_path: Path):
    """Aggregates metrics per method and topic, saves to CSV, and prints table."""
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    methods = ["no_switch", "scripted", "soft_prefix"]

    aggregated_rows = []

    for method in methods:
        m_records = [r for r in records if r["method"] == method]
        for topic_id in range(10):
            t_records = [r for r in m_records if r["topic_id"] == topic_id]
            if not t_records:
                continue
            topic_name = t_records[0]["topic_name"]
            n = len(t_records)
            b_acc = sum(1 for r in t_records if r["b_correct"]) / n
            c_acc = sum(1 for r in t_records if r["c_correct"]) / n
            mean_reward = sum(r["reward"] for r in t_records) / n
            mean_ppl = sum(r["ppl"] for r in t_records) / n
            gate_block_rate = sum(1 for r in t_records if r["gate_blocked"]) / n

            aggregated_rows.append({
                "method": method,
                "topic_id": topic_id,
                "topic_name": topic_name,
                "b_acc": round(b_acc, 3),
                "c_acc": round(c_acc, 3),
                "reward": round(mean_reward, 3),
                "ppl": round(mean_ppl, 2),
                "gate_block_rate": round(gate_block_rate, 3)
            })

    # Save to CSV
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "topic_id", "topic_name", "b_acc", "c_acc", "reward", "ppl", "gate_block_rate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregated_rows)

    print(f"\n[Saved] Aggregated comparison table saved to {out_csv_path}")

    # Print Summary Table
    print("\n" + "=" * 90)
    print(f"{'Method':<15} | {'B-Acc (%)':<10} | {'C-Acc (%)':<10} | {'Mean Reward':<12} | {'Mean PPL':<10} | {'Gate Block Rate (%)':<18}")
    print("-" * 90)
    for method in methods:
        m_rows = [r for r in aggregated_rows if r["method"] == method]
        avg_b = sum(r["b_acc"] for r in m_rows) / len(m_rows) * 100
        avg_c = sum(r["c_acc"] for r in m_rows) / len(m_rows) * 100
        avg_rew = sum(r["reward"] for r in m_rows) / len(m_rows)
        avg_ppl = sum(r["ppl"] for r in m_rows) / len(m_rows)
        avg_block = sum(r["gate_block_rate"] for r in m_rows) / len(m_rows) * 100
        print(f"{method:<15} | {avg_b:<10.1f} | {avg_c:<10.1f} | {avg_rew:<12.3f} | {avg_ppl:<10.2f} | {avg_block:<18.1f}")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_per_topic", type=int, default=20, help="Runs per topic per method")
    args = parser.parse_args()

    runner = BenchmarkRunner()
    records = runner.run_all(runs_per_topic=args.runs_per_topic)

    # Save raw records
    raw_path = runner.config.RESULTS_DIR / "benchmark_raw_records.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    # Save aggregated table
    csv_path = runner.config.RESULTS_DIR / "comparison_table.csv"
    aggregate_and_save(records, csv_path)
