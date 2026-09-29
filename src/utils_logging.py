import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

class ExperimentLogger:
    def __init__(self, config):
        self.config = config
        self.wandb_run = None
        self.log_history = []

        if self.config.ENABLE_WANDB:
            try:
                import wandb
                wandb.login(key=self.config.WANDB_API_KEY)
                self.wandb_run = wandb.init(
                    project=self.config.WANDB_PROJECT,
                    entity=self.config.WANDB_ENTITY if self.config.WANDB_ENTITY else None,
                    config={
                        "base_model": self.config.BASE_MODEL_NAME,
                        "device": self.config.DEVICE,
                        "num_sensitive_topics": self.config.NUM_SENSITIVE_TOPICS,
                        "num_cover_topics": self.config.NUM_COVER_TOPICS,
                        "pretrain_epochs": self.config.PRETRAIN_EPOCHS,
                        "pretrain_lr": self.config.PRETRAIN_LR,
                    }
                )
                print(f"[W&B] Initialized run: {self.wandb_run.name} in project {self.config.WANDB_PROJECT}")
            except Exception as e:
                print(f"[W&B Warning] Failed to initialize wandb ({e}). Falling back to local logging.")
                self.wandb_run = None
        else:
            print("[Logger] W&B disabled or no API key provided. Logging metrics locally.")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to wandb and store locally."""
        entry = {"step": step, **metrics}
        self.log_history.append(entry)

        if self.wandb_run is not None:
            import wandb
            wandb.log(metrics, step=step)

    def save_local_results(self, filename: str, data: Dict[str, Any]):
        """Save a dictionary as a formatted JSON file in results directory."""
        filepath = self.config.RESULTS_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[Logger] Saved results to {filepath}")

    def finish(self):
        if self.wandb_run is not None:
            import wandb
            wandb.finish()
            print("[W&B] Run completed and synced.")

def push_checkpoint_to_hf(checkpoint_path: Path, config, commit_message: str = "Upload model checkpoint"):
    """Optionally push saved checkpoints to Hugging Face Hub."""
    if not config.ENABLE_HF_PUSH or not config.HF_REPO_ID:
        print("[HuggingFace] HF push skipped (HF_TOKEN or HF_REPO_ID not configured).")
        return

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=config.HF_TOKEN)
        print(f"[HuggingFace] Pushing {checkpoint_path} to {config.HF_REPO_ID}...")
        api.create_repo(repo_id=config.HF_REPO_ID, repo_type="model", exist_ok=True)
        if checkpoint_path.is_file():
            api.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=checkpoint_path.name,
                repo_id=config.HF_REPO_ID,
                repo_type="model",
                commit_message=commit_message
            )
        elif checkpoint_path.is_dir():
            api.upload_folder(
                folder_path=str(checkpoint_path),
                repo_id=config.HF_REPO_ID,
                repo_type="model",
                commit_message=commit_message
            )
        print(f"[HuggingFace] Successfully uploaded to https://huggingface.co/{config.HF_REPO_ID}")
    except Exception as e:
        print(f"[HuggingFace Error] Failed to push to Hugging Face Hub: {e}")
