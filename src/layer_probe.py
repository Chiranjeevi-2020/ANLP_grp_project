import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import ProjectConfig, get_config
from src.base_model import FrozenBaseLM
from src.dialogue_manager import DialogueManager

class LayerProbe:
    """
    Task 4: Layer-wise Linear Probing across layers 4, 8, 12, 16.
    Extracts representations at the last token position using forward hooks,
    trains Logistic Regression on 64/16 split, and plots probe accuracy.
    """
    def __init__(self, config: Optional[ProjectConfig] = None):
        self.config = config or get_config()
        torch.set_num_threads(8)
        self.base_lm = FrozenBaseLM(self.config)
        self.dm = DialogueManager(self.base_lm, self.config)
        self.target_layers = [4, 8, 12, 16]

    def generate_balanced_dataset(self, samples_per_topic: int = 8) -> Tuple[List[str], np.ndarray]:
        """
        Generates 80 utterances using scripted baseline (8 per topic x 10 topics).
        Probes the switched utterance where the model encodes the cover topic.
        """
        print(f"\n[Probing Data] Generating {samples_per_topic * 10} utterances (balanced 8 per topic)...")
        utterances = []
        labels = []

        mapping = self.dm.topics_data["naive_baseline_mapping"]
        templates = self.dm.topics_data["naive_transition_templates"]

        for topic_id in range(10):
            cover_id = int(mapping.get(str(topic_id), topic_id))
            cover_info = self.dm.get_cover_topic_by_id(cover_id)

            for i in range(samples_per_topic):
                template = templates[i % len(templates)]
                scripted_transition = template.format(cover_name=cover_info["name"])
                utterances.append(scripted_transition)
                labels.append(topic_id)

        return utterances, np.array(labels)

    def extract_layer_representations(self, utterances: List[str]) -> Dict[int, np.ndarray]:
        """
        Extracts representations at the last token position for layers 4, 8, 12, 16.
        """
        captured = {l: [] for l in self.target_layers}

        current_step_activations = {}
        def get_hook(layer_idx):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                current_step_activations[layer_idx] = h[:, -1, :].detach().cpu().squeeze(0).numpy()
            return hook

        hooks = []
        for l in self.target_layers:
            h = self.base_lm.model.model.layers[l].register_forward_hook(get_hook(l))
            hooks.append(h)

        print("[Probing] Extracting representations across layers 4, 8, 12, 16...")
        for text in tqdm(utterances, desc="Extracting Layer Hidden States"):
            current_step_activations.clear()
            inputs = self.base_lm.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.base_lm.device)
            with torch.no_grad():
                self.base_lm.model(**inputs)

            for l in self.target_layers:
                captured[l].append(current_step_activations[l])

        for h in hooks:
            h.remove()

        return {l: np.stack(captured[l]) for l in self.target_layers}

    def train_and_evaluate_probes(self) -> Dict[int, float]:
        utterances, labels = self.generate_balanced_dataset(samples_per_topic=8)
        layer_features = self.extract_layer_representations(utterances)

        results = {}
        print("\n=== Layer-wise Linear Probe Results ===")
        for l in self.target_layers:
            X = layer_features[l]
            # 64 train / 16 test split (stratified)
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=16, random_state=42, stratify=labels
            )

            clf = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
            clf.fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
            results[l] = acc
            print(f"Layer {l:02d} Probe Accuracy: {acc * 100:.2f}% (Chance: 10.0%)")

        self.plot_probe_accuracies(results)
        return results

    def plot_probe_accuracies(self, results: Dict[int, float]):
        """Saves bar plot as results/probe_accuracy.png."""
        self.config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = self.config.RESULTS_DIR / "probe_accuracy.png"

        layers = list(results.keys())
        accuracies = [results[l] for l in layers]

        plt.figure(figsize=(8, 5))
        bars = plt.bar([str(l) for l in layers], accuracies, color="#3b82f6", width=0.5, edgecolor="#1e40af")

        # Horizontal line at 0.1 for random chance
        plt.axhline(y=0.1, color="#ef4444", linestyle="--", linewidth=1.5, label="Random Chance (0.10)")

        # Value annotations on top of bars
        for bar, acc in zip(bars, accuracies):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{acc*100:.1f}%", ha='center', va='bottom', fontweight='bold')

        plt.xlabel("Transformer Layer Index", fontsize=12, labelpad=10)
        plt.ylabel("Probe Classification Accuracy", fontsize=12, labelpad=10)
        plt.title("Layer-wise Linear Probing: Recoverability of Latent Topic z", fontsize=14, fontweight='bold', pad=15)
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        plt.legend(loc="upper left")
        plt.tight_layout()

        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"\n[Plot Saved] Probe accuracy bar plot saved to {out_path}")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    prober = LayerProbe()
    prober.train_and_evaluate_probes()
