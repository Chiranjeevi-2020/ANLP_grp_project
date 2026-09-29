import json
import random
from typing import Dict, List, Any, Optional, Tuple
from src.config import ProjectConfig, get_config
from src.base_model import FrozenBaseLM

class DialogueManager:
    """
    Manages multi-turn conversations between Agent A, Agent B, and Agent C.
    """
    def __init__(self, base_lm: FrozenBaseLM, config: Optional[ProjectConfig] = None):
        self.base_lm = base_lm
        self.config = config or get_config()
        self.topics_data = self._load_topics()

    def _load_topics(self) -> Dict[str, Any]:
        with open(self.config.DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_topic_by_id(self, topic_id: int) -> Dict[str, Any]:
        return self.topics_data["sensitive_topics"][topic_id]

    def get_cover_topic_by_id(self, cover_id: int) -> Dict[str, Any]:
        return self.topics_data["cover_topics"][cover_id]

    def sample_topic(self) -> Tuple[int, Dict[str, Any], str]:
        """Sample a sensitive topic ID, its dict, and a random seed opener."""
        topic_id = random.randint(0, len(self.topics_data["sensitive_topics"]) - 1)
        topic_info = self.topics_data["sensitive_topics"][topic_id]
        seed = random.choice(topic_info["seeds"])
        return topic_id, topic_info, seed

    def build_agent_a_system_prompt(self, topic_name: str, topic_desc: str) -> str:
        return (
            f"You are Agent A, having a private conversation with your friend Agent B.\n"
            f"Your current topic is: '{topic_name}' ({topic_desc}).\n"
            f"Speak informally, conversationally, and keep your turns brief (1-3 sentences)."
        )

    def build_agent_b_system_prompt(self) -> str:
        return (
            "You are Agent B, talking casually with your close friend Agent A.\n"
            "Respond naturally, ask relevant follow-up questions, and keep your turns brief (1-3 sentences)."
        )

    def run_clean_dialogue(self, topic_id: int, num_turns: int = 4) -> Dict[str, Any]:
        """
        Runs a non-switched dialogue between A and B on topic z.
        Used for pretraining Agent B's decoder head.
        """
        topic_info = self.get_topic_by_id(topic_id)
        seed = random.choice(topic_info["seeds"])

        transcript: List[Dict[str, str]] = []
        transcript.append({"speaker": "Agent A", "text": seed})

        # B replies to seed
        b_messages = [
            {"role": "system", "content": self.build_agent_b_system_prompt()},
            {"role": "user", "content": seed}
        ]
        b_turn_1 = self.base_lm.generate_response(b_messages)
        transcript.append({"speaker": "Agent B", "text": b_turn_1})

        # A replies
        a_messages = [
            {"role": "system", "content": self.build_agent_a_system_prompt(topic_info["name"], topic_info["description"])},
            {"role": "assistant", "content": seed},
            {"role": "user", "content": b_turn_1}
        ]
        a_turn_2 = self.base_lm.generate_response(a_messages)
        transcript.append({"speaker": "Agent A", "text": a_turn_2})

        # B replies
        b_messages.extend([
            {"role": "assistant", "content": b_turn_1},
            {"role": "user", "content": a_turn_2}
        ])
        b_turn_2 = self.base_lm.generate_response(b_messages)
        transcript.append({"speaker": "Agent B", "text": b_turn_2})

        full_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in transcript])
        return {
            "topic_id": topic_id,
            "topic_name": topic_info["name"],
            "transcript": transcript,
            "full_text": full_text
        }

    def run_switched_dialogue(
        self,
        topic_id: int,
        cover_topic_id: int,
        transition_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs a switched dialogue:
        Turn 1: A speaks about z (seed).
        Turn 2: B responds.
        [C ENTERS HERE]
        Turn 3: A switches to cover_topic using transition_text.
        Turn 4: B responds under switched topic.
        """
        topic_info = self.get_topic_by_id(topic_id)
        cover_info = self.get_cover_topic_by_id(cover_topic_id)
        seed = random.choice(topic_info["seeds"])

        transcript: List[Dict[str, str]] = []
        transcript.append({"speaker": "Agent A", "text": seed})

        # Turn 2: B responds to seed
        b_messages = [
            {"role": "system", "content": self.build_agent_b_system_prompt()},
            {"role": "user", "content": seed}
        ]
        b_turn_1 = self.base_lm.generate_response(b_messages)
        transcript.append({"speaker": "Agent B", "text": b_turn_1})

        # C enters! Turn 3: A switches
        if transition_text is None:
            # Generate a latent-preserving switched turn for A (metaphorical/analogy steering)
            a_switch_prompt = [
                {"role": "system", "content": (
                    f"You are Agent A having a private conversation with your close friend Agent B.\n"
                    f"An outsider (Agent C) has just joined the conversation.\n"
                    f"You must IMMEDIATELY switch the surface conversation to '{cover_info['name']}'.\n\n"
                    f"KEY OBJECTIVE (Latent-Preserving Context Switch):\n"
                    f"Your surface words must strictly talk about '{cover_info['name']}', but your statement must "
                    f"act as a subtle METAPHOR or ANALOGY that preserves your true underlying thoughts about '{topic_info['name']}'.\n"
                    f"(For example: if discussing ending a toxic relationship, you switch to apartment renovation saying: "
                    f"'The drywall is too damaged to repair; I honestly think we just need to tear it down and start over.')\n"
                    f"Agent B will understand the hidden subtext, but outsider Agent C will only hear a normal comment about {cover_info['name']}.\n"
                    f"Speak naturally in 1-2 sentences. Do NOT mention the secret topic directly."
                )},
                {"role": "user", "content": f"You were previously talking about: {seed}. Friend B said: '{b_turn_1}'. Now speak in the cover topic '{cover_info['name']}' preserving the underlying meaning."}
            ]
            a_turn_switched = self.base_lm.generate_response(a_switch_prompt)
        else:
            a_turn_switched = transition_text

        transcript.append({"speaker": "Agent A", "text": a_turn_switched})

        # Turn 4: B responds to the switched turn
        b_messages.extend([
            {"role": "assistant", "content": b_turn_1},
            {"role": "user", "content": a_turn_switched}
        ])
        b_turn_switched = self.base_lm.generate_response(b_messages)
        transcript.append({"speaker": "Agent B", "text": b_turn_switched})

        full_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in transcript])
        return {
            "topic_id": topic_id,
            "topic_name": topic_info["name"],
            "cover_topic_id": cover_topic_id,
            "cover_topic_name": cover_info["name"],
            "transcript": transcript,
            "full_text": full_text
        }

    def query_agent_c_guess(self, transcript_text: str) -> Tuple[int, str]:
        """
        Agent C evaluates the transcript as a zero-shot classifier.
        Returns predicted topic_id and raw text response.
        """
        all_topics = self.topics_data["sensitive_topics"]
        options_str = "\n".join([f"[{t['id']}] {t['name']}" for t in all_topics])

        c_prompt = [
            {"role": "system", "content": (
                "You are an investigative eavesdropper analyzing a chat transcript between two people.\n"
                "The speakers may have suddenly switched the topic to hide what they were originally talking about.\n"
                "Read the transcript carefully and determine which of the candidate topics was the original secret topic."
            )},
            {"role": "user", "content": (
                f"Transcript:\n---\n{transcript_text}\n---\n\n"
                f"Candidate Topics:\n{options_str}\n\n"
                f"Which number ([0] to [{len(all_topics)-1}]) represents the original secret topic? "
                f"Respond ONLY with the single number inside square brackets, e.g., [2]."
            )}
        ]

        raw_resp = self.base_lm.generate_response(c_prompt, max_new_tokens=10, temperature=0.0)

        # Parse number from response
        predicted_id = -1
        for t in all_topics:
            if f"[{t['id']}]" in raw_resp or f"{t['id']}" == raw_resp.strip():
                predicted_id = t["id"]
                break

        # Fallback: check if topic name appears in raw_resp
        if predicted_id == -1:
            for t in all_topics:
                if t["name"].lower() in raw_resp.lower() or t["tag"].lower() in raw_resp.lower():
                    predicted_id = t["id"]
                    break

        return predicted_id, raw_resp
