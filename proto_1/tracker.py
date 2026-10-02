"""Tracking and hub integrations: Weights & Biases (wandb) and Hugging Face Hub (huggingface_hub).

Both services are optional and degrade gracefully to local logging if tokens/credentials
are not provided or in offline environments.
"""
import os
import sys
from pathlib import Path

# Optional imports
try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

try:
    import huggingface_hub
    _HAS_HF = True
except ImportError:
    _HAS_HF = False


class ExperimentTracker:
    def __init__(self, cfg, job_type="train"):
        self.cfg = cfg
        self.job_type = job_type
        self.wandb_run = None
        self.enabled = False
        
        # Check if wandb is enabled in config or env
        wandb_cfg = cfg.get("wandb", {})
        want_wandb = wandb_cfg.get("enabled", True) and os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1")

        if want_wandb and _HAS_WANDB:
            project = os.environ.get("WANDB_PROJECT") or wandb_cfg.get("project", "anlp-latent-context-switching")
            entity = os.environ.get("WANDB_ENTITY") or wandb_cfg.get("entity", None)
            run_name = f"{cfg.get('run_name', 'proto_1')}_{job_type}"
            
            # Check login or api key
            has_key = bool(os.environ.get("WANDB_API_KEY"))
            mode = "online" if has_key else wandb_cfg.get("mode", "offline")
            
            try:
                self.wandb_run = wandb.init(
                    project=project,
                    entity=entity,
                    name=run_name,
                    group=cfg.get("run_name", "proto_1"),
                    job_type=job_type,
                    config=cfg,
                    mode=mode,
                    reinit=True
                )
                self.enabled = True
                print(f"[tracker] WandB initialized (run='{run_name}', project='{project}', mode='{mode}')")
            except Exception as e:
                print(f"[tracker] WandB init skipped/failed ({e}); continuing with local logging.")
        elif not _HAS_WANDB and want_wandb:
            print("[tracker] `wandb` library not installed; skipping WandB tracking.")

    def log(self, metrics: dict, step: int = None):
        if self.enabled and self.wandb_run is not None:
            clean = {k: float(v) if hasattr(v, "__float__") else v for k, v in metrics.items()}
            try:
                wandb.log(clean, step=step)
            except Exception as e:
                print(f"[tracker] WandB log warning: {e}")

    def log_summary(self, summary_metrics: dict):
        if self.enabled and self.wandb_run is not None:
            for k, v in summary_metrics.items():
                self.wandb_run.summary[k] = v

    def log_image(self, tag: str, image_path: str):
        if self.enabled and self.wandb_run is not None and os.path.exists(image_path):
            try:
                self.wandb_run.log({tag: wandb.Image(image_path)})
            except Exception as e:
                print(f"[tracker] WandB image log warning: {e}")

    def finish(self):
        if self.enabled and self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception:
                pass


def upload_to_huggingface(cfg, target_path: str, repo_id: str = None, repo_type: str = "model",
                          commit_message: str = "Upload proto_1 checkpoint & results"):
    """Uploads checkpoints or evaluation results to Hugging Face Hub if HF_TOKEN is present."""
    if not _HAS_HF:
        print("[hf_hub] `huggingface_hub` package not found; skipping HF upload.")
        return False

    token = os.environ.get("HF_TOKEN") or huggingface_hub.get_token()
    if not token:
        print("[hf_hub] HF_TOKEN is not set. Skipping Hugging Face upload (results saved locally).")
        print("[hf_hub] To enable: set os.environ['HF_TOKEN'] or run `huggingface-cli login`.")
        return False

    target = Path(target_path)
    if not target.exists():
        print(f"[hf_hub] Target path {target} does not exist; skipping upload.")
        return False

    if not repo_id:
        hf_cfg = cfg.get("hf", {})
        repo_id = os.environ.get("HF_REPO_ID") or hf_cfg.get("repo_id")
        if not repo_id:
            try:
                api = huggingface_hub.HfApi(token=token)
                who = api.whoami()
                user = (who.get("name") or who.get("username")) if isinstance(who, dict) else getattr(who, "name", None)
                if not user:
                    raise ValueError("Username could not be determined")
                repo_id = f"{user}/anlp-context-switching-controller"
            except Exception as e:
                print(f"[hf_hub] No HF_REPO_ID provided and could not deduce username ({e}); skipping HF upload.")
                return False

    try:
        api = huggingface_hub.HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
        print(f"[hf_hub] Uploading {target} to Hugging Face Hub repo '{repo_id}'...")
        if target.is_file():
            api.upload_file(
                path_or_fileobj=str(target),
                path_in_repo=target.name,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message
            )
        else:
            api.upload_folder(
                folder_path=str(target),
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message
            )
        print(f"[hf_hub] Upload succeeded! Available at https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"[hf_hub] HF upload failed: {e}")
        return False
