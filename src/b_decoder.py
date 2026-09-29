import torch
import torch.nn as nn
from typing import Tuple

class BDecoderHead(nn.Module):
    """
    Lightweight classification head attached to Agent B.
    Takes B's contextual dialogue embedding from the frozen base LLM
    and predicts the secret topic z in {0, ..., num_sensitive_topics - 1}.
    """
    def __init__(
        self,
        base_hidden_dim: int,
        num_sensitive_topics: int = 10,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(base_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_sensitive_topics)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, context_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_embedding: [batch_size, base_hidden_dim]
        Returns:
            logits: [batch_size, num_sensitive_topics]
        """
        return self.classifier(context_embedding)

    def predict_topic(self, context_embedding: torch.Tensor) -> Tuple[int, float]:
        """
        Predicts single topic ID and softmax probability score.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(context_embedding)
            probs = torch.softmax(logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_id].item()
        return pred_id, confidence

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    head = BDecoderHead(base_hidden_dim=896, num_sensitive_topics=10)
    print(f"[B Decoder Test] Trainable parameters: {head.count_parameters():,}")
    dummy_input = torch.randn(4, 896)
    logits = head(dummy_input)
    print("Logits shape:", logits.shape)
    assert logits.shape == (4, 10), "Logits shape mismatch!"
    print("[B Decoder Test] Passed successfully!")
