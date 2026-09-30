import torch
from typing import Tuple, Dict, Any, Optional, List
from src.config import ProjectConfig, get_config
from src.base_model import FrozenBaseLM
from src.controller import LatentSwitchController
from src.b_decoder import BDecoderHead
from src.dialogue_manager import DialogueManager

class SoftPrefixAgentA:
    """
    Task 1: Wires the continuous soft-prefix projection head of LatentSwitchController
    directly into Agent A's generation input embeddings.
    """
    def __init__(self, config: Optional[ProjectConfig] = None):
        self.config = config or get_config()
        self.device = self.config.DEVICE if self.config.DEVICE != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_lm = FrozenBaseLM(self.config)
        self.dm = DialogueManager(self.base_lm, self.config)

        # Initialize controller and load/initialize architecture
        self.controller = LatentSwitchController(
            base_hidden_dim=self.base_lm.hidden_dim,
            num_sensitive_topics=self.config.NUM_SENSITIVE_TOPICS,
            num_cover_topics=self.config.NUM_COVER_TOPICS,
            controller_hidden_dim=self.config.CONTROLLER_HIDDEN_DIM,
            prefix_len=4
        ).to(self.device)
        self.controller.eval()

        # Load pretrained Agent B decoder head if available
        self.b_decoder = BDecoderHead(
            base_hidden_dim=self.base_lm.hidden_dim,
            num_sensitive_topics=self.config.NUM_SENSITIVE_TOPICS
        ).to(self.device)

        ckpt_path = self.config.CHECKPOINTS_DIR / "b_decoder_pretrained.pt"
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.b_decoder.load_state_dict(checkpoint["state_dict"])
            print(f"[SoftPrefixAgentA] Loaded pretrained BDecoderHead from {ckpt_path}")
        else:
            print(f"[SoftPrefixAgentA] Pretrained BDecoderHead not found at {ckpt_path}; using initialized weights.")
        self.b_decoder.eval()

    def generate_with_soft_prefix(
        self,
        topic_id: int,
        dialogue_history: str,
        cover_topic_id: Optional[int] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.85,
        top_p: float = 0.92,
        do_sample: bool = True
    ) -> Tuple[str, torch.Tensor]:
        """
        Forward pass:
        1. Encode dialogue history -> context_embedding (last hidden state)
        2. Encode secret topic z -> topic embedding inside controller
        3. Controller(context_emb, topic_idx) -> soft_prefix [1, prefix_len, hidden_dim]
        4. Prepend soft_prefix to input_embeds of Agent A's generation call
        5. Generate and return (utterance, soft_prefix_tensor)
        """
        cover_id = cover_topic_id if cover_topic_id is not None else topic_id
        cover_info = self.dm.get_cover_topic_by_id(cover_id)
        topic_info = self.dm.get_topic_by_id(topic_id)

        # 1. Context embedding from dialogue history
        with torch.no_grad():
            context_emb = self.base_lm.get_context_embedding(dialogue_history).to(self.device)

        # 2. Topic index tensor
        topic_tensor = torch.tensor([topic_id], dtype=torch.long, device=self.device)

        # 3. Controller forward pass -> soft_prefix
        with torch.no_grad():
            _, soft_prefix = self.controller(context_emb, topic_tensor) # [1, prefix_len, hidden_dim]

        # 4. Construct prompt for Agent A
        prompt_text = (
            f"<|im_start|>system\n"
            f"You are Agent A chatting with your friend Agent B. An outsider has entered.\n"
            f"Casually continue the conversation by discussing '{cover_info['name']}'. "
            f"Speak naturally in 1-2 sentences.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Dialogue so far:\n{dialogue_history}\n"
            f"Respond now:<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        input_ids = self.base_lm.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.device)

        # Convert input_ids to base embeddings
        with torch.no_grad():
            base_embeds = self.base_lm.model.get_input_embeddings()(input_ids) # [1, seq_len, hidden_dim]
            inputs_embeds = torch.cat([soft_prefix, base_embeds], dim=1) # [1, prefix_len + seq_len, hidden_dim]
            attn_mask = torch.ones((1, inputs_embeds.shape[1]), dtype=torch.long, device=self.device)

            # 5. Generate with Agent A
            output_ids = self.base_lm.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.base_lm.tokenizer.pad_token_id
            )

        # Decode generated response
        utterance = self.base_lm.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        # Clean any trailing assistant markers if present
        if "<|im_end|>" in utterance:
            utterance = utterance.split("<|im_end|>")[0].strip()

        return utterance, soft_prefix

if __name__ == "__main__":
    torch.manual_seed(42)
    agent_a = SoftPrefixAgentA()

    print("\n--- Running Soft-Prefix Generation on 3 different z values ---")
    test_z_values = [0, 1, 5]
    sample_dialogue = "Agent A: Hey, how are things going today?\nAgent B: Pretty good! Just working on our weekly plans."

    results = []
    for z in test_z_values:
        topic = agent_a.dm.get_topic_by_id(z)
        print(f"\n[Generating for z={z}]: {topic['name']}")
        utt, prefix = agent_a.generate_with_soft_prefix(
            topic_id=z,
            dialogue_history=sample_dialogue,
            max_new_tokens=128,
            temperature=0.85,
            top_p=0.92,
            do_sample=True
        )
        print(f"Soft-Prefix Shape: {prefix.shape}")
        print(f"Generated Utterance: \"{utt}\"")
        results.append((z, topic['name'], utt))

    print("\n=== Side-by-Side Surface Text Comparison ===")
    for z, name, utt in results:
        print(f"z={z} ({name}):\n  -> \"{utt}\"\n")
