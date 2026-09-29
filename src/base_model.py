import torch
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import ProjectConfig, get_config

class FrozenBaseLM:
    """
    Wrapper for the frozen pretrained base language model.
    Runs locally on CPU or GPU/NPU. Base model weights are never updated.
    """
    def __init__(self, config: Optional[ProjectConfig] = None):
        self.config = config or get_config()
        self.device = self._resolve_device()
        print(f"[Model Loader] Initializing base model: {self.config.BASE_MODEL_NAME} on device: {self.device}")

        # Select data type based on device
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.BASE_MODEL_NAME,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.BASE_MODEL_NAME,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(self.device)

        # Freeze all base model parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.hidden_dim = self.model.config.hidden_size
        print(f"[Model Loader] Base model ready (hidden_dim = {self.hidden_dim}, parameters frozen: True)")

    def _resolve_device(self) -> str:
        if self.config.DEVICE != "auto":
            return self.config.DEVICE
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @torch.no_grad()
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        Generate an instruction-tuned conversational response using chat templates.
        """
        max_new_tokens = max_new_tokens or self.config.MAX_NEW_TOKENS
        temperature = temperature or self.config.TEMPERATURE
        top_p = top_p or self.config.TOP_P

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        do_sample = temperature > 0.0
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id
        )

        # Decode only the generated response tokens
        generated_tokens = output_ids[0, inputs.input_ids.shape[1]:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return response_text

    @torch.no_grad()
    def get_context_embedding(self, text: str) -> torch.Tensor:
        """
        Extract the contextual representation (last hidden state of final token)
        for a given text string. Shape: [1, hidden_dim].
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        # Last hidden state of the last token
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [1, hidden_dim]
        return last_hidden

if __name__ == "__main__":
    # Self-test when run directly
    base = FrozenBaseLM()
    test_msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you in 1 sentence?"}
    ]
    resp = base.generate_response(test_msgs)
    print("Test Generation:", resp)
    emb = base.get_context_embedding("Hello world")
    print("Embedding shape:", emb.shape)
