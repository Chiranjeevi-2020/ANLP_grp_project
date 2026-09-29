import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

class LatentSwitchController(nn.Module):
    """
    Lightweight trainable controller module (1-5M parameters).
    Observes dialogue context embedding + secret topic z, and outputs
    a conditioning signal for Agent A (either a discrete cover topic choice
    or soft-prompt tokens).
    """
    def __init__(
        self,
        base_hidden_dim: int,
        num_sensitive_topics: int = 10,
        num_cover_topics: int = 10,
        controller_hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        prefix_len: int = 4
    ):
        super().__init__()
        self.base_hidden_dim = base_hidden_dim
        self.num_sensitive_topics = num_sensitive_topics
        self.num_cover_topics = num_cover_topics
        self.prefix_len = prefix_len

        # Learnable embedding for the secret topic z
        self.topic_emb_dim = 128
        self.topic_embedding = nn.Embedding(num_sensitive_topics, self.topic_emb_dim)

        # Input dimension: LLM context embedding + topic embedding
        in_dim = base_hidden_dim + self.topic_emb_dim

        # Backbone MLP
        layers = []
        current_dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, controller_hidden_dim))
            layers.append(nn.LayerNorm(controller_hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            current_dim = controller_hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Head 1: Discrete cover topic classifier / policy logits
        self.discrete_head = nn.Linear(controller_hidden_dim, num_cover_topics)

        # Head 2: Continuous soft-prompt projection (prefix_len * base_hidden_dim)
        self.soft_prompt_head = nn.Linear(controller_hidden_dim, prefix_len * base_hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        context_emb: torch.Tensor,
        topic_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            context_emb: [batch_size, base_hidden_dim]
            topic_idx: [batch_size] containing secret topic IDs (0..9)
        Returns:
            discrete_logits: [batch_size, num_cover_topics]
            soft_prompts: [batch_size, prefix_len, base_hidden_dim]
        """
        batch_size = context_emb.size(0)
        t_emb = self.topic_embedding(topic_idx) # [batch_size, topic_emb_dim]
        combined = torch.cat([context_emb, t_emb], dim=-1)

        features = self.backbone(combined) # [batch_size, controller_hidden_dim]

        discrete_logits = self.discrete_head(features) # [batch_size, num_cover_topics]

        soft_prompts_flat = self.soft_prompt_head(features)
        soft_prompts = soft_prompts_flat.view(batch_size, self.prefix_len, self.base_hidden_dim)

        return discrete_logits, soft_prompts

    def sample_cover_topic(
        self,
        context_emb: torch.Tensor,
        topic_idx: torch.Tensor
    ) -> Tuple[int, torch.Tensor]:
        """
        Samples a cover topic index and returns its log probability (for REINFORCE in Week 5).
        """
        discrete_logits, _ = self.forward(context_emb, topic_idx)
        dist = torch.distributions.Categorical(logits=discrete_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test Controller shape and parameter counts
    hidden_dim = 896  # Qwen2.5-0.5B hidden dim is 896 (Qwen2.5-1.5B is 1536)
    controller = LatentSwitchController(base_hidden_dim=hidden_dim)
    print(f"[Controller Test] Total trainable parameters: {controller.count_parameters():,}")

    dummy_context = torch.randn(2, hidden_dim)
    dummy_topics = torch.tensor([0, 3], dtype=torch.long)
    d_logits, s_prompts = controller(dummy_context, dummy_topics)
    print("Discrete logits shape:", d_logits.shape)
    print("Soft prompts shape:", s_prompts.shape)
    assert d_logits.shape == (2, 10), "Logits shape mismatch!"
    assert s_prompts.shape == (2, 4, hidden_dim), "Soft prompt shape mismatch!"
    print("[Controller Test] Passed successfully!")
