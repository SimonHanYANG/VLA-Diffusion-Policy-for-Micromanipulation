"""CLIP text encoder for fixed language instructions.

Each task has one fixed instruction. We precompute CLIP embeddings once
and cache them, so the full CLIP model is not needed during training.
"""

import torch
import torch.nn as nn


class TextEncoder:
    """Encodes task language instructions using CLIP.

    This is a standalone encoder (not nn.Module) because CLIP is only
    used once to precompute embeddings before training.
    """

    TASK_INSTRUCTIONS = {
        "microsphere": "Navigate the microsphere to the red laser dot.",
        "yeast": "Navigate the yeast cell to the red laser dot.",
        "sperm_head": "Navigate the sperm head to the red laser dot.",
        "sperm_tail": "Navigate the sperm tail tip to the red laser dot.",
    }

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        if self._model is None:
            from transformers import CLIPModel, CLIPTokenizer
            self._tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import CLIPTokenizer
            self._tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """Return (B, 512) text embeddings."""
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        outputs = self.model.get_text_features(**inputs)
        if isinstance(outputs, torch.Tensor):
            return outputs.cpu()
        # transformers >= 4.40 returns BaseModelOutputWithPooling
        return outputs.pooler_output.cpu()

    def get_instruction(self, task_name: str) -> str:
        return self.TASK_INSTRUCTIONS.get(task_name, f"Navigate the {task_name} to the red laser dot.")

    def precompute_all(self, output_path: str | None = None) -> dict[str, torch.Tensor]:
        """Precompute embeddings for all known tasks."""
        embeddings = {}
        for task_name, instruction in self.TASK_INSTRUCTIONS.items():
            emb = self.encode([instruction])  # (1, 512)
            embeddings[task_name] = emb.squeeze(0)  # (512,)

        if output_path is not None:
            import os
            parent = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(parent, exist_ok=True)
            torch.save(embeddings, output_path)

        return embeddings


class CachedTextEmbeddings:
    """Loads and serves precomputed CLIP text embeddings for training."""

    def __init__(self, cache_path: str | None = None, embeddings: dict | None = None):
        if cache_path is not None:
            import os
            if os.path.exists(cache_path):
                self.embeddings = torch.load(cache_path, map_location="cpu", weights_only=True)
            else:
                self.embeddings = {}
        elif embeddings is not None:
            self.embeddings = embeddings
        else:
            self.embeddings = {}

    def get(self, task_name: str, default: torch.Tensor | None = None) -> torch.Tensor:
        if task_name in self.embeddings:
            return self.embeddings[task_name]
        if default is not None:
            return default
        raise KeyError(f"No embedding for task '{task_name}'")

    def __getitem__(self, task_name: str) -> torch.Tensor:
        return self.embeddings[task_name]

    def __contains__(self, task_name: str) -> bool:
        return task_name in self.embeddings
