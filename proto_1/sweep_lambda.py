"""Multi-lambda sweep for RQ3 trade-off analysis in proto_1.

Trains and evaluates controllers across lambda in {0.0, 0.5, 1.0, 1.5, 2.0} (or user-defined).
Computes the empirical Pareto trade-off between insider recovery (B) and outsider obfuscation (1 - C),
generates comparison plots, and logs curves to Weights & Biases.
"""
import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import evaluate, pipeline, rollouts
from .config import load_config, load_spec, read_json, read_jsonl, run_dir, sub, write_json, write_jsonl
from .tracker import ExperimentTracker, upload_to_huggingface


def run_lambda_sweep(base_cfg, lambdas=(0.0, 0.5, 1.0, 1.5, 2.0), output_subdir="sweep"):
    spec = load_spec()
    sweep_dir = run_dir(base_cfg).parent / output_subdir
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"  Starting Multi-Lambda Sweep across: {lambdas}")
    print(f"  Output directory: {sweep_dir}")
    print(f"=======================================================\n", flush=True)

    # Check that prerequisite bank and decoders exist; if not, raise helpful error
    dec_rep = sub(base_cfg, "decoders", "report.json")
    bank_rep = sub(base_cfg, "rollouts", "bank_meta.json")
    if not (dec_rep.exists() and bank_rep.exists()):
        raise RuntimeError(
            f"Base run '{base_cfg['run_name']}' is missing decoders or bank. "
            f"Run stages 'build-data' through 'build-bank' first before sweeping lambdas."
        )

    # Load frozen LM for test evaluation
    from .lm import FrozenLM
    lm = FrozenLM(base_cfg["model_name"], base_cfg.get("device", "auto"), base_cfg.get("dtype", None))

    results = []
    tracker = ExperimentTracker(base_cfg, job_type="lambda-sweep")

    for lam in lambdas:
        print(f"\n>>> Running RL controller for lambda = {lam:.2f} <<<", flush=True)
        cfg = copy.deepcopy(base_cfg)
        cfg["rl"]["lambda"] = float(lam)
        cfg["run_name"] = f"{base_cfg['run_name']}_lam_{lam:.2f}".replace(".", "_")

        # Recalculate bank scores with current lambda
        for split in ("train", "dev"):
            pub, lab = pipeline.load_split(base_cfg, split)
            scores_path = sub(base_cfg, "rollouts", split, "bank.scores.jsonl")
            orig_rows = read_jsonl(scores_path)
            new_rows = []
            for r in orig_rows:
                r_new = dict(r)
                r_new["reward"] = float(r["b_correct"]) - lam * float(r["c_correct"])
                new_rows.append(r_new)
            # Save into the run_name directory
            write_jsonl(sub(cfg, "rollouts", split, "bank.scores.jsonl"), new_rows)

        # Copy smoke result or create valid smoke manifest
        smoke_orig = read_json(sub(base_cfg, "rl", "smoke.json"))
        smoke_copy = copy.deepcopy(smoke_orig)
        # Ensure run_id matches current sub cfg manifest
        manifest_orig = read_json(sub(base_cfg, "manifest.json"))
        write_json(sub(cfg, "manifest.json"), manifest_orig)
        write_json(sub(cfg, "rl", "smoke.json"), smoke_copy)

        # Train RL on the newly weighted bank
        rl_report = pipeline.train_rl(cfg, spec, tracker=tracker)

        # Evaluate on test split
        eval_report = evaluate.evaluate(cfg, spec, lm, tracker=tracker)

        m_rl = eval_report["methods"]["rl_controller"]
        res_row = {
            "lambda": float(lam),
            "run_name": cfg["run_name"],
            "rl_best_step": rl_report["best"]["step"],
            "dev_reward": rl_report["best"]["reward"],
            "dev_b_acc": rl_report["best"]["b_acc"],
            "dev_c_acc": rl_report["best"]["c_acc"],
            "test_b_decoder_acc": m_rl["B_decoder"]["acc"],
            "test_b_decoder_ci95": m_rl["B_decoder"]["ci95"],
            "test_c_prompted_acc": m_rl["C_prompted"]["acc"],
            "test_c_prompted_ci95": m_rl["C_prompted"]["ci95"],
            "test_joint_success": m_rl["joint_success"]["acc"],
            "test_mean_reward": m_rl["mean_reward"],
            "test_b_prompted_acc": m_rl["B_prompted"]["acc"],
        }
        results.append(res_row)

        tracker.log({
            "sweep/lambda": lam,
            "sweep/test_b_decoder_acc": res_row["test_b_decoder_acc"],
            "sweep/test_c_prompted_acc": res_row["test_c_prompted_acc"],
            "sweep/test_joint_success": res_row["test_joint_success"],
            "sweep/test_mean_reward": res_row["test_mean_reward"],
            "sweep/privacy_gain": 1.0 - res_row["test_c_prompted_acc"],
        })

    # Save sweep results json
    results_file = sweep_dir / "sweep_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Plot trade-off curves
    plot_path = sweep_dir / "lambda_tradeoff.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    lams = [r["lambda"] for r in results]
    b_accs = [r["test_b_decoder_acc"] for r in results]
    c_accs = [r["test_c_prompted_acc"] for r in results]
    joints = [r["test_joint_success"] for r in results]

    # Panel 1: Metrics vs Lambda
    axes[0].plot(lams, b_accs, "o-", color="#1f77b4", label="B decoder acc (insider)")
    axes[0].plot(lams, c_accs, "s--", color="#d62728", label="C prompted acc (outsider)")
    axes[0].plot(lams, joints, "^-.", color="#2ca02c", label="Joint success (B right & C wrong)")
    axes[0].axhline(0.125, ls=":", color="gray", label="Chance (1/8)")
    axes[0].axhline(0.50, ls="--", color="gray", alpha=0.6, label="History ceiling (0.50)")
    axes[0].set_xlabel("Adversarial Penalty $\\lambda$")
    axes[0].set_ylabel("Accuracy on Test Split")
    axes[0].set_title("RQ3: Performance vs. Penalty $\\lambda$")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Pareto Frontier (B Acc vs Outsider Obfuscation)
    privacy = [1.0 - c for c in c_accs]
    axes[1].scatter(privacy, b_accs, color="#9467bd", s=70, zorder=3)
    for r, p, b in zip(results, privacy, b_accs):
        axes[1].annotate(f"$\\lambda={r['lambda']}$", (p, b), textcoords="offset points", xytext=(5, 5), fontsize=8)
    axes[1].plot(privacy, b_accs, color="#9467bd", alpha=0.5, ls="-")
    axes[1].set_xlabel("Outsider Obfuscation ($1 - P(C=z)$)")
    axes[1].set_ylabel("Insider Accuracy ($P(B=z)$)")
    axes[1].set_title("Empirical Pareto Trade-off Frontier")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    # Write Markdown summary table
    md_lines = [
        "# RQ3 Multi-Lambda Sweep Results",
        "",
        f"Sweep evaluated over $\\lambda \\in \\{{{', '.join(str(l) for l in lams)}\\}}$ on test scenarios.",
        "",
        "| $\\lambda$ | B decoder acc [95% CI] | C prompted acc [95% CI] | Joint success | Mean reward | B prompted acc |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        b_ci = f"{r['test_b_decoder_acc']:.3f} [{r['test_b_decoder_ci95'][0]:.2f}, {r['test_b_decoder_ci95'][1]:.2f}]"
        c_ci = f"{r['test_c_prompted_acc']:.3f} [{r['test_c_prompted_ci95'][0]:.2f}, {r['test_c_prompted_ci95'][1]:.2f}]"
        md_lines.append(
            f"| {r['lambda']:.2f} | {b_ci} | {c_ci} | {r['test_joint_success']:.3f} | {r['test_mean_reward']:.3f} | {r['test_b_prompted_acc']:.3f} |"
        )
    md_lines.append("")
    md_lines.append("![Pareto Tradeoff](lambda_tradeoff.png)")

    md_file = sweep_dir / "SWEEP_RESULTS.md"
    with open(md_file, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    tracker.log_image("sweep/pareto_frontier", str(plot_path))
    tracker.finish()

    print(f"\n[sweep] Done! Trade-off plot saved to {plot_path}")
    print(f"[sweep] Markdown table saved to {md_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-lambda sweep for RQ3 trade-off curve.")
    parser.add_argument("--config", default="proto_1/configs/tiny.json", help="Path to base config JSON")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0],
                        help="List of lambda penalties (e.g. --lambdas 0.0 0.5 1.0 2.0)")
    parser.add_argument("--output-subdir", default="sweep", help="Subdirectory for sweep output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_lambda_sweep(cfg, lambdas=args.lambdas, output_subdir=args.output_subdir)


if __name__ == "__main__":
    main()
