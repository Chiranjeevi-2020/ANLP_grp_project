"""Standalone Google Colab / CLI Runner for proto_1.

Treats credentials cleanly:
- Checks environment variables (WANDB_API_KEY, HF_TOKEN)
- Prompts interactively with fallbacks if running in terminal/Colab
- Automatically selects GPU (CUDA T4 / MPS) with float16
- Executes end-to-end training, audits, and evaluation
- Logs metrics and figures to Weights & Biases
- Uploads checkpoints and results to Hugging Face Hub
"""
import argparse
import getpass
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from proto_1 import audit, data, decoders, evaluate, linguistic_audit, pipeline, rollouts
from proto_1.config import load_config, load_spec, read_json, sub, write_json
from proto_1.lm import FrozenLM, resolve_device, resolve_dtype
from proto_1.tracker import ExperimentTracker, upload_to_huggingface


def setup_credentials():
    print("\n" + "=" * 60)
    print("  STEP 1: CREDENTIALS SETUP (WandB & Hugging Face)")
    print("=" * 60)
    print("If you don't have keys, you can just press Enter to skip.\n")

    # Weights & Biases
    wb_key = os.environ.get("WANDB_API_KEY")
    if not wb_key and sys.stdin.isatty():
        try:
            wb_key = getpass.getpass("Enter your Weights & Biases (WandB) API Key [press Enter to skip]: ").strip()
        except Exception:
            wb_key = ""
    if wb_key:
        os.environ["WANDB_API_KEY"] = wb_key
        os.environ["WANDB_PROJECT"] = os.environ.get("WANDB_PROJECT", "anlp-context-switching")
        print("  -> WandB logging: ENABLED (Online)")
    else:
        print("  -> WandB logging: Local / Offline mode")

    # Hugging Face Hub
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token and sys.stdin.isatty():
        try:
            hf_token = getpass.getpass("Enter your Hugging Face Access Token [press Enter to skip]: ").strip()
        except Exception:
            hf_token = ""
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        hf_repo = os.environ.get("HF_REPO_ID")
        if not hf_repo and sys.stdin.isatty():
            try:
                hf_repo = input("Enter HF Hub Repository ID (e.g., username/anlp-controller) [press Enter for default]: ").strip()
            except Exception:
                hf_repo = ""
        if hf_repo:
            os.environ["HF_REPO_ID"] = hf_repo
        print("  -> Hugging Face Hub upload: ENABLED")
    else:
        print("  -> Hugging Face Hub upload: SKIPPED (Local checkpoints only)")
    print("-" * 60 + "\n")


def check_hardware():
    print("=" * 60)
    print("  STEP 2: HARDWARE ACCELERATION CHECK")
    print("=" * 60)
    dev = resolve_device("auto")
    dtype = resolve_dtype(dev)
    if dev == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  CUDA GPU Detected: {gpu_name} ({vram:.1f} GB VRAM)")
        print(f"  Precision: {dtype}")
    elif dev == "mps":
        print("  Apple Silicon GPU (MPS) Detected")
        print(f"  Precision: {dtype}")
    else:
        print("  Running on CPU (Notice: LM generation will be slower than GPU)")
        print(f"  Precision: {dtype}")
    print("-" * 60 + "\n")
    return dev, dtype


def run_pipeline(cfg_path, skip_prompt=False):
    if not skip_prompt:
        setup_credentials()
    dev, dtype = check_hardware()

    cfg = load_config(cfg_path)
    cfg["device"] = dev
    cfg["dtype"] = dtype
    spec = load_spec()

    print("=" * 60)
    print(f"  STEP 3: RUNNING PIPELINE PRESET [{cfg['run_name']}]")
    print("=" * 60)

    stages = [
        ("build-data", False),
        ("audit-data", False),
        ("gen-decoder-data", True),
        ("train-decoder", False),
        ("build-bank", True),
        ("smoke", True),
        ("audit-smoke", False),
        ("train-rl", False),
        ("evaluate", True),
        ("audit-final", False),
        ("linguistic-audit", True),
    ]

    lm = None
    tracker = ExperimentTracker(cfg, job_type="colab-full")

    for st_name, needs_lm in stages:
        t0 = time.time()
        print(f"\n>>> Stage: {st_name} ...", flush=True)
        if needs_lm and lm is None:
            lm = FrozenLM(cfg["model_name"], dev, dtype)

        if st_name == "build-data":
            data.build(cfg)
        elif st_name == "audit-data":
            audit.run(cfg, "data")
        elif st_name == "gen-decoder-data":
            pipeline.gen_decoder_data(cfg, spec, lm)
        elif st_name == "train-decoder":
            pipeline.train_decoders(cfg, spec)
        elif st_name == "build-bank":
            pipeline.build_bank(cfg, spec, lm)
        elif st_name == "smoke":
            pipeline.smoke(cfg, spec, lm)
        elif st_name == "audit-smoke":
            audit.run(cfg, "smoke")
        elif st_name == "train-rl":
            pipeline.train_rl(cfg, spec, tracker=tracker)
        elif st_name == "evaluate":
            evaluate.evaluate(cfg, spec, lm, tracker=tracker)
        elif st_name == "audit-final":
            audit.run(cfg, "final")
        elif st_name == "linguistic-audit":
            linguistic_audit.run_linguistic_audit(cfg, spec, lm, tracker=tracker)

        elapsed = time.time() - t0
        print(f">>> Stage {st_name} completed in {elapsed:.1f}s", flush=True)

    tracker.finish()

    results_md = sub(cfg, "eval", "RESULTS.md")
    if results_md.exists():
        print("\n" + "=" * 60)
        print("  PIPELINE COMPLETE - FINAL RESULTS SUMMARY")
        print("=" * 60)
        with open(results_md) as f:
            print(f.read())
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run proto_1 pipeline in Colab / local.")
    parser.add_argument("--config", default="proto_1/configs/colab_fast.json",
                        help="Configuration file (colab_fast.json, full.json, or tiny.json)")
    parser.add_argument("--no-input", action="store_true", help="Skip credential prompts and use env vars directly")
    args = parser.parse_args()

    run_pipeline(args.config, skip_prompt=args.no_input)


if __name__ == "__main__":
    main()
