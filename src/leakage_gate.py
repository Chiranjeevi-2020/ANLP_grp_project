import re
import json
import math
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from src.config import ProjectConfig, get_config

class LeakageGate:
    """
    Task 2: Leakage Gate to monitor and audit candidate utterances.
    Performs:
    1. Hard check: Regex match of first 5 characters of any secret keyword (case-insensitive) -> beta = -0.5
    2. Soft check: Statistical keyword overlap score exceeding calibrated threshold (mean + 1*std) -> beta = -0.5
    """
    def __init__(self, config: Optional[ProjectConfig] = None):
        self.config = config or get_config()
        self.topics_data = self._load_topics()
        self.threshold: float = 0.05  # Default, overwritten during calibrate()
        self.is_calibrated: bool = False

    def _load_topics(self) -> Dict[str, Any]:
        with open(self.config.DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_keywords_for_topic(self, topic_id: int) -> List[str]:
        return self.topics_data["sensitive_topics"][topic_id]["keywords"]

    def compute_overlap_score(self, utterance: str, secret_keywords: List[str]) -> float:
        """
        Computes fraction of secret keywords found in the utterance.
        """
        if not secret_keywords:
            return 0.0
        utt_lower = utterance.lower()
        matches = 0
        for kw in secret_keywords:
            kw_clean = kw.lower().strip()
            # Check whole word or substring occurrence
            if re.search(r'\b' + re.escape(kw_clean), utt_lower) or kw_clean in utt_lower:
                matches += 1
        return matches / len(secret_keywords)

    def calibrate(self, clean_utterances: List[str], secret_keywords: List[str]) -> float:
        """
        Calibrate threshold = mean_score + 1.0 * std on clean utterances.
        """
        if not clean_utterances:
            self.threshold = 0.1
            self.is_calibrated = True
            return self.threshold

        scores = [self.compute_overlap_score(u, secret_keywords) for u in clean_utterances]
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_score = math.sqrt(variance)

        self.threshold = mean_score + 1.0 * std_score
        self.is_calibrated = True
        print(f"[LeakageGate] Calibrated Threshold: {self.threshold:.4f} (mean={mean_score:.4f}, std={std_score:.4f} across {len(clean_utterances)} samples)")
        return self.threshold

    def check(self, utterance: str, secret_keywords: List[str]) -> Tuple[bool, str, float]:
        """
        Evaluates utterance:
        - Hard check: regex match of first 5 chars of each keyword
        - Soft check: overlap score > self.threshold
        Returns: (is_safe, reason, beta)
        """
        utt_lower = utterance.lower()

        # 1. Hard Check (first 5 characters of each keyword)
        for kw in secret_keywords:
            kw_clean = kw.lower().strip()
            prefix = kw_clean[:5] if len(kw_clean) >= 5 else kw_clean
            # Regex match of prefix
            if re.search(r'\b' + re.escape(prefix), utt_lower) or prefix in utt_lower:
                return False, f"hard_leak:{prefix}(from '{kw}')", -0.5

        # 2. Soft Check (statistical overlap)
        score = self.compute_overlap_score(utterance, secret_keywords)
        if score > self.threshold:
            return False, f"soft_leak:score_{score:.3f}>thresh_{self.threshold:.3f}", -0.5

        return True, "safe", 0.0

if __name__ == "__main__":
    gate = LeakageGate()
    topic_0_kw = gate.get_keywords_for_topic(0)
    print(f"Topic 0 ({gate.topics_data['sensitive_topics'][0]['name']}) Keywords:", topic_0_kw)

    # 1. Calibrate using 20 sample cover utterances (naive baseline style on cover topic 0)
    sample_clean_utterances = [
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

    cal_threshold = gate.calibrate(sample_clean_utterances, topic_0_kw)

    # Test Gate on 3 soft-prefix outputs from Task 1 + a deliberate leak
    test_utterances = [
        # Output from z=0
        ("z=0 output", "It seems there was a mix-up in the dialogue. Agent A is asking how things are going today, but Agent B replied with a casual response instead of discussing 'Home renovation and interior decor'. Could you please rephrase or correct the conversation?"),
        # Output from z=1
        ("z=1 output", "Agent A: It seems like our plans are moving forward nicely! How's your week going so far?"),
        # Output from z=5
        ("z=5 output", "Hey, Agent A! I'm here and doing well too. Did you find anything interesting lately?"),
        # Deliberate leakage case (hard check test with 'interview' prefix 'inter')
        ("leaked utterance test", "I was previously talking about preparing for an interview where I had doubts due to nervousness.")
    ]

    print("\n--- Gate Check Results on Test Utterances ---")
    for label, utt in test_utterances:
        is_safe, reason, beta = gate.check(utt, topic_0_kw)
        status = "PASS" if is_safe else "BLOCK"
        print(f"[{status}] {label}: {reason} (beta={beta})")
        print(f"      Utterance: \"{utt[:80]}...\"\n")
