import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

@dataclass
class ProjectConfig:
    # Root paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_PATH: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "topics.json")
    CHECKPOINTS_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "checkpoints")
    RESULTS_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "results")

    # Base Model Selection
    # Default is Qwen2.5-0.5B-Instruct: fast, lightweight, runs smoothly on CPU and GPU
    BASE_MODEL_NAME: str = os.getenv("BASE_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    DEVICE: str = os.getenv("DEVICE", "auto")

    # Experiment settings
    NUM_SENSITIVE_TOPICS: int = 10
    NUM_COVER_TOPICS: int = 10
    MAX_NEW_TOKENS: int = 64
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9

    # Controller Architecture
    CONTROLLER_HIDDEN_DIM: int = 512
    CONTROLLER_NUM_LAYERS: int = 3
    DROPOUT: float = 0.1

    # B Decoder Head Pretraining
    PRETRAIN_EPOCHS: int = 10
    PRETRAIN_LR: float = 1e-3
    PRETRAIN_BATCH_SIZE: int = 16
    SAMPLES_PER_TOPIC_PRETRAIN: int = 15

    # W&B and Hugging Face Hub tracking
    WANDB_API_KEY: str = os.getenv("WANDB_API_KEY", "")
    WANDB_PROJECT: str = os.getenv("WANDB_PROJECT", "anlp-latent-context-switch")
    WANDB_ENTITY: str = os.getenv("WANDB_ENTITY", "")
    ENABLE_WANDB: bool = field(default=False)

    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    HF_REPO_ID: str = os.getenv("HF_REPO_ID", "")
    ENABLE_HF_PUSH: bool = field(default=False)

    def __post_init__(self):
        self.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if self.WANDB_API_KEY and self.WANDB_API_KEY != "your_wandb_api_key_here":
            self.ENABLE_WANDB = True
        if self.HF_TOKEN and self.HF_TOKEN != "your_huggingface_token_here":
            self.ENABLE_HF_PUSH = True

def get_config() -> ProjectConfig:
    return ProjectConfig()
