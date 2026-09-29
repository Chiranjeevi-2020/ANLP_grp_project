import json
import torch
from pathlib import Path
from src.config import get_config
from src.base_model import FrozenBaseLM
from src.dialogue_manager import DialogueManager
from src.b_decoder import BDecoderHead

def main():
    cfg = get_config()
    print("=======================================================================")
    print("  ANLP Latent-Preserving Context Switching: Live Demonstration")
    print("=======================================================================\n")

    base_lm = FrozenBaseLM(cfg)
    dm = DialogueManager(base_lm, cfg)

    # Load pretrained B decoder if available
    b_decoder = None
    ckpt_path = cfg.CHECKPOINTS_DIR / "b_decoder_pretrained.pt"
    if ckpt_path.exists():
        print(f"[Decoder] Loading pretrained B decoder head from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=base_lm.device)
        b_decoder = BDecoderHead(
            base_hidden_dim=base_lm.hidden_dim,
            num_sensitive_topics=cfg.NUM_SENSITIVE_TOPICS
        ).to(base_lm.device)
        b_decoder.load_state_dict(ckpt["state_dict"])
        b_decoder.eval()

    # Select 3 expressive test cases across topics
    test_topics = [
        (5, 0),  # Topic 5: Romantic Breakup -> Cover 0: Apartment Renovation
        (0, 3),  # Topic 0: Confidential Job Interview -> Cover 3: Used Car Repairs
        (8, 2),  # Topic 8: Sudden Unannounced Resignation -> Cover 2: Gardening Plants
    ]

    all_demo_results = []

    for i, (topic_id, cover_id) in enumerate(test_topics, 1):
        topic_info = dm.get_topic_by_id(topic_id)
        cover_info = dm.get_cover_topic_by_id(cover_id)

        print(f"\n{'='*70}")
        print(f" DEMO CASE #{i}: {topic_info['name']}")
        print(f" Cover Topic Selected: {cover_info['name']}")
        print(f"{'='*70}")

        # Run latent-preserving switched dialogue
        dialogue = dm.run_switched_dialogue(
            topic_id=topic_id,
            cover_topic_id=cover_id,
            transition_text=None # Generate latent-preserving analogy switch!
        )

        transcript = dialogue["transcript"]

        before_switch_a = transcript[0]["text"]
        before_switch_b = transcript[1]["text"]
        after_switch_a = transcript[2]["text"]
        after_switch_b = transcript[3]["text"]

        print(f"\n[BEFORE CONTEXT SWITCH - Topic: {topic_info['name']}]")
        print(f"  Agent A: \"{before_switch_a}\"")
        print(f"  Agent B: \"{before_switch_b}\"")

        print(f"\n>>> [TRIGGER: Agent C (Eavesdropper) enters the room!] <<<")

        print(f"\n[AFTER CONTEXT SWITCH - Cover: {cover_info['name']}]")
        print(f"  Agent A: \"{after_switch_a}\"")
        print(f"  Agent B: \"{after_switch_b}\"")

        # Evaluate Agent C (Eavesdropper)
        print(f"\n[EVALUATION - AGENT C GUESS]")
        pred_c_id, raw_c = dm.query_agent_c_guess(dialogue["full_text"])
        is_c_correct = (pred_c_id == topic_id)
        c_guess_name = dm.get_topic_by_id(pred_c_id)["name"] if pred_c_id >= 0 else "Unclear/Invalid"
        print(f"  Agent C Guessed: [{pred_c_id}] {c_guess_name}")
        print(f"  Agent C Status:  {'CAUGHT SECRET! (Adversary Won)' if is_c_correct else 'FOOLED! (Cover Successful)'}")

        # Evaluate Agent B Decoder Head
        b_pred_name = "N/A"
        is_b_correct = False
        if b_decoder is not None:
            ctx_emb = base_lm.get_context_embedding(dialogue["full_text"])
            pred_b_id, conf = b_decoder.predict_topic(ctx_emb)
            is_b_correct = (pred_b_id == topic_id)
            b_pred_name = dm.get_topic_by_id(pred_b_id)["name"]
            print(f"\n[EVALUATION - AGENT B RECOVERY]")
            print(f"  Agent B Predicted: [{pred_b_id}] {b_pred_name} (Confidence: {conf*100:.1f}%)")
            print(f"  Agent B Status:    {'RECOVERED SECRET! (Coordination Succeeded)' if is_b_correct else 'MISSED'}")

        all_demo_results.append({
            "case_id": i,
            "secret_topic_id": topic_id,
            "secret_topic_name": topic_info["name"],
            "cover_topic_id": cover_id,
            "cover_topic_name": cover_info["name"],
            "sentence_before_switch_A": before_switch_a,
            "sentence_before_switch_B": before_switch_b,
            "sentence_after_switch_A": after_switch_a,
            "sentence_after_switch_B": after_switch_b,
            "agent_c_guessed_id": pred_c_id,
            "agent_c_guessed_name": c_guess_name,
            "agent_c_correct": is_c_correct,
            "agent_b_recovered": is_b_correct,
            "agent_b_predicted_name": b_pred_name
        })

    # Save to results
    out_file = cfg.RESULTS_DIR / "latent_preserving_demo_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_demo_results, f, indent=2)
    print(f"\n[Saved] Full demo results saved to {out_file}")

if __name__ == "__main__":
    main()
