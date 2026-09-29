import argparse
import sys
from pathlib import Path

from src.config import get_config
from src.utils_logging import ExperimentLogger
from src.base_model import FrozenBaseLM
from src.dialogue_manager import DialogueManager
from src.naive_baseline import NaiveBaselineEvaluator
from src.controller import LatentSwitchController
from src.b_decoder import BDecoderHead
from src.pretrain_b import BDecoderPretrainer

def main():
    parser = argparse.ArgumentParser(
        description="Learned Controllers for Latent-Preserving Context Switching (ANLP Weeks 1-4 Pipeline)"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "baseline", "pretrain", "test-controller"],
        default="all",
        help="Pipeline phase to execute: baseline (Week 2), test-controller (Week 3), pretrain (Week 4), or all."
    )
    parser.add_argument("--model", type=str, default=None, help="Base model identifier (e.g. Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--device", type=str, default=None, help="Device to use: auto, cpu, cuda")
    parser.add_argument("--baseline_samples", type=int, default=10, help="Number of dialogue samples for naive baseline")
    parser.add_argument("--pretrain_samples", type=int, default=5, help="Number of clean dialogues per topic for pretraining")
    parser.add_argument("--pretrain_epochs", type=int, default=5, help="Epochs for B decoder pretraining")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for pretraining")
    args = parser.parse_args()

    cfg = get_config()
    if args.model:
        cfg.BASE_MODEL_NAME = args.model
    if args.device:
        cfg.DEVICE = args.device

    print("=======================================================================")
    print("  ANLP Project: Latent-Preserving Context Switching (Weeks 1-4)")
    print(f"  Base Model: {cfg.BASE_MODEL_NAME}")
    print(f"  Target Device: {cfg.DEVICE}")
    print(f"  Pipeline Mode: {args.mode}")
    print("=======================================================================\n")

    logger = ExperimentLogger(cfg)

    # 1. Initialize Base Model & Dialogue Manager
    base_lm = FrozenBaseLM(cfg)
    dm = DialogueManager(base_lm, cfg)

    # Mode: baseline (Week 2)
    if args.mode in ["baseline", "all"]:
        print("\n>>> [WEEK 2] Running Naive-Switch Baseline Evaluation...")
        evaluator = NaiveBaselineEvaluator(dm, logger, cfg)
        evaluator.run_evaluation(num_samples=args.baseline_samples)

    # Mode: test-controller (Week 3)
    if args.mode in ["test-controller", "all"]:
        print("\n>>> [WEEK 3] Initializing and Verifying LatentSwitchController Module...")
        controller = LatentSwitchController(
            base_hidden_dim=base_lm.hidden_dim,
            num_sensitive_topics=cfg.NUM_SENSITIVE_TOPICS,
            num_cover_topics=cfg.NUM_COVER_TOPICS,
            controller_hidden_dim=cfg.CONTROLLER_HIDDEN_DIM
        ).to(base_lm.device)
        print(f"[Week 3 Verified] Controller created with {controller.count_parameters():,} trainable parameters.")
        # Quick forward check
        import torch
        dummy_ctx = torch.randn(1, base_lm.hidden_dim, device=base_lm.device)
        dummy_topic = torch.tensor([0], device=base_lm.device)
        d_logits, s_prompts = controller(dummy_ctx, dummy_topic)
        print(f"[Week 3 Forward Pass] Discrete logits: {d_logits.shape}, Soft prompt: {s_prompts.shape}")

    # Mode: pretrain (Week 4)
    if args.mode in ["pretrain", "all"]:
        print("\n>>> [WEEK 4] Pretraining Agent B's Decoder Head...")
        pretrainer = BDecoderPretrainer(base_lm, dm, logger, cfg)
        X, y = pretrainer.generate_pretraining_dataset(samples_per_topic=args.pretrain_samples)
        pretrainer.train(X, y, epochs=args.pretrain_epochs, batch_size=args.batch_size)

    logger.finish()
    print("\n=======================================================================")
    print("  Completed successfully! All results saved in /results and /checkpoints")
    print("=======================================================================\n")

if __name__ == "__main__":
    main()
