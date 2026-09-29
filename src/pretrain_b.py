import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm

from src.config import ProjectConfig, get_config
from src.base_model import FrozenBaseLM
from src.dialogue_manager import DialogueManager
from src.b_decoder import BDecoderHead
from src.utils_logging import ExperimentLogger, push_checkpoint_to_hf

class BDecoderPretrainer:
    """
    Week 4 Pipeline:
    Pretrains Agent B's Decoder Head on clean (non-switched) multi-turn dialogues
    so the controller receives a stable signal during RL training.
    """
    def __init__(
        self,
        base_lm: FrozenBaseLM,
        dialogue_manager: DialogueManager,
        logger: ExperimentLogger,
        config: ProjectConfig
    ):
        self.base_lm = base_lm
        self.dm = dialogue_manager
        self.logger = logger
        self.config = config
        self.device = self.base_lm.device

        self.b_head = BDecoderHead(
            base_hidden_dim=self.base_lm.hidden_dim,
            num_sensitive_topics=self.config.NUM_SENSITIVE_TOPICS
        ).to(self.device)

    def generate_pretraining_dataset(self, samples_per_topic: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates clean non-switched dialogues and extracts B's context embeddings.
        """
        print(f"\n[Pretraining Data] Generating {samples_per_topic} clean dialogues per topic...")
        all_embeddings = []
        all_labels = []

        total_topics = len(self.dm.topics_data["sensitive_topics"])
        for topic_id in range(total_topics):
            for _ in range(samples_per_topic):
                dialogue = self.dm.run_clean_dialogue(topic_id=topic_id, num_turns=4)
                # Context embedding of the full clean dialogue
                emb = self.base_lm.get_context_embedding(dialogue["full_text"]) # [1, hidden_dim]
                all_embeddings.append(emb.squeeze(0).cpu())
                all_labels.append(topic_id)

        X = torch.stack(all_embeddings) # [N, hidden_dim]
        y = torch.tensor(all_labels, dtype=torch.long)
        print(f"[Pretraining Data] Generated dataset of shape X: {X.shape}, y: {y.shape}")
        return X, y

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        epochs = epochs or self.config.PRETRAIN_EPOCHS
        lr = lr or self.config.PRETRAIN_LR
        batch_size = batch_size or self.config.PRETRAIN_BATCH_SIZE

        # Train/Validation split (80/20)
        n = X.size(0)
        indices = torch.randperm(n)
        split = int(0.8 * n)
        train_idx, val_idx = indices[:split], indices[split:]

        train_dataset = TensorDataset(X[train_idx], y[train_idx])
        val_dataset = TensorDataset(X[val_idx], y[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.b_head.parameters(), lr=lr, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        history = []

        print(f"\n[Pretraining B Decoder] Starting training for {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            self.b_head.train()
            total_train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.b_head(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=-1)
                train_correct += (preds == batch_y).sum().item()
                train_total += batch_x.size(0)

            train_loss = total_train_loss / max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)

            # Validation
            self.b_head.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    logits = self.b_head(batch_x)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
                    preds = torch.argmax(logits, dim=-1)
                    val_correct += (preds == batch_y).sum().item()
                    val_total += batch_x.size(0)

            val_acc = val_correct / max(val_total, 1) if val_total > 0 else 0.0
            avg_val_loss = val_loss / max(val_total, 1) if val_total > 0 else 0.0

            epoch_metrics = {
                "pretrain/epoch": epoch,
                "pretrain/train_loss": train_loss,
                "pretrain/train_acc": train_acc,
                "pretrain/val_loss": avg_val_loss,
                "pretrain/val_acc": val_acc
            }
            history.append(epoch_metrics)
            self.logger.log(epoch_metrics, step=epoch)

            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                # Save best checkpoint
                ckpt_path = self.config.CHECKPOINTS_DIR / "b_decoder_pretrained.pt"
                torch.save({
                    "state_dict": self.b_head.state_dict(),
                    "base_hidden_dim": self.base_lm.hidden_dim,
                    "num_sensitive_topics": self.config.NUM_SENSITIVE_TOPICS,
                    "best_val_acc": best_val_acc,
                    "epoch": epoch
                }, ckpt_path)

        print(f"\n[Pretraining Done] Best Validation Accuracy: {best_val_acc * 100:.1f}%")
        best_ckpt_path = self.config.CHECKPOINTS_DIR / "b_decoder_pretrained.pt"
        print(f"[Checkpoints] Saved to {best_ckpt_path}")

        # Push to Hugging Face if enabled
        push_checkpoint_to_hf(
            checkpoint_path=best_ckpt_path,
            config=self.config,
            commit_message=f"Pretrained B-Decoder (Val Acc: {best_val_acc*100:.1f}%)"
        )

        results = {
            "best_val_accuracy": best_val_acc,
            "epochs": epochs,
            "training_history": history
        }
        self.logger.save_local_results("b_decoder_pretraining_results.json", results)
        return results

if __name__ == "__main__":
    cfg = get_config()
    logger = ExperimentLogger(cfg)
    base_lm = FrozenBaseLM(cfg)
    dm = DialogueManager(base_lm, cfg)
    pretrainer = BDecoderPretrainer(base_lm, dm, logger, cfg)
    X, y = pretrainer.generate_pretraining_dataset(samples_per_topic=2)
    pretrainer.train(X, y, epochs=3, batch_size=4)
    logger.finish()
