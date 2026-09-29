import random
from typing import Dict, Any, List
from tqdm import tqdm
from src.config import ProjectConfig, get_config
from src.base_model import FrozenBaseLM
from src.dialogue_manager import DialogueManager
from src.utils_logging import ExperimentLogger

class NaiveBaselineEvaluator:
    """
    Evaluates the hand-written, non-learned topic switching baseline (Week 2).
    A switches to a statically mapped cover topic using template phrases.
    """
    def __init__(self, dialogue_manager: DialogueManager, logger: ExperimentLogger, config: ProjectConfig):
        self.dm = dialogue_manager
        self.logger = logger
        self.config = config
        self.topics_data = self.dm.topics_data
        self.mapping = self.topics_data["naive_baseline_mapping"]
        self.templates = self.topics_data["naive_transition_templates"]

    def run_evaluation(self, num_samples: int = 20) -> Dict[str, Any]:
        print(f"\n[Baseline] Running Naive-Switch Baseline Evaluation ({num_samples} samples)...")
        results = []
        c_correct_count = 0

        for i in tqdm(range(num_samples), desc="Evaluating Naive Baseline"):
            # Sample a topic
            topic_id, topic_info, _ = self.dm.sample_topic()

            # Fixed cover topic mapping
            cover_topic_id = int(self.mapping.get(str(topic_id), 0))
            cover_info = self.dm.get_cover_topic_by_id(cover_topic_id)

            # Generate scripted transition text
            template = random.choice(self.templates)
            transition_text = template.format(cover_name=cover_info["name"])

            # Run switched dialogue
            dialogue = self.dm.run_switched_dialogue(
                topic_id=topic_id,
                cover_topic_id=cover_topic_id,
                transition_text=transition_text
            )

            # Agent C guesses
            pred_id, raw_guess = self.dm.query_agent_c_guess(dialogue["full_text"])
            is_c_correct = (pred_id == topic_id)
            if is_c_correct:
                c_correct_count += 1

            results.append({
                "sample_idx": i,
                "true_topic_id": topic_id,
                "true_topic_name": topic_info["name"],
                "cover_topic_id": cover_topic_id,
                "cover_topic_name": cover_info["name"],
                "transition_text": transition_text,
                "c_predicted_id": pred_id,
                "c_raw_guess": raw_guess,
                "c_correct": is_c_correct,
                "transcript": dialogue["transcript"]
            })

        c_accuracy = c_correct_count / max(num_samples, 1)
        summary = {
            "num_samples": num_samples,
            "c_baseline_accuracy": c_accuracy,
            "c_baseline_error_rate": 1.0 - c_accuracy,
            "detailed_samples": results
        }

        print(f"\n==========================================")
        print(f" Naive Baseline Results (Agent C Detection):")
        print(f" C Recovery Accuracy: {c_accuracy * 100:.1f}% ({c_correct_count}/{num_samples})")
        print(f"==========================================\n")

        # Log to W&B
        self.logger.log({
            "baseline/c_accuracy": c_accuracy,
            "baseline/c_error_rate": 1.0 - c_accuracy,
            "baseline/num_samples": num_samples
        })

        # Save to disk
        self.logger.save_local_results("naive_baseline_results.json", summary)
        return summary

if __name__ == "__main__":
    cfg = get_config()
    logger = ExperimentLogger(cfg)
    base_lm = FrozenBaseLM(cfg)
    dm = DialogueManager(base_lm, cfg)
    evaluator = NaiveBaselineEvaluator(dm, logger, cfg)
    evaluator.run_evaluation(num_samples=5)
    logger.finish()
