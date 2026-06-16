import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_EMBEDDING_MODEL_ID = "hfl/chinese-macbert-base"
DEFAULT_EMBEDDING_MODEL_DIR = "./models/chinese-macbert-base"
ENCODER_ARTIFACT_DIR = "macbert_encoder"
EMBEDDING_DIM = 768


def get_main_repo_root(root_dir):
    """Return the main repo root when running inside a git worktree."""
    if ".claude/worktrees" in root_dir:
        return os.path.dirname(os.path.dirname(os.path.dirname(root_dir)))
    return root_dir


def resolve_relative_to_root(path, root_dir):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(root_dir, path))


def get_base_model_dir(root_dir):
    model_dir = os.getenv("EMBEDDING_MODEL_DIR", DEFAULT_EMBEDDING_MODEL_DIR)
    return resolve_relative_to_root(model_dir, root_dir)


def resolve_encoder_path(root_dir, prefer_artifact=True):
    """Find the frozen MacBERT encoder artifact, falling back to the base model."""
    main_repo_root = get_main_repo_root(root_dir)
    candidates = []
    if prefer_artifact:
        candidates.extend([
            os.path.join(root_dir, "my_final_six_intents_model", ENCODER_ARTIFACT_DIR),
            os.path.join(main_repo_root, "my_final_six_intents_model", ENCODER_ARTIFACT_DIR),
        ])
    candidates.extend([
        get_base_model_dir(root_dir),
        get_base_model_dir(main_repo_root),
    ])

    return next((path for path in candidates if os.path.exists(path)), None)


class MacBertEncoder:
    """Frozen MacBERT [CLS] encoder with a SentenceTransformer-like encode API."""

    def __init__(self, model_path, device="cpu", max_length=128, batch_size=32):
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        if self.embedding_dim != EMBEDDING_DIM:
            raise ValueError(
                f"MacBERT 编码器维度应为 {EMBEDDING_DIM}，但 {model_path} 的 hidden_size={self.embedding_dim}。"
                "请检查 .env 中的 EMBEDDING_MODEL_ID/EMBEDDING_MODEL_DIR 是否仍指向旧 BGE 模型。"
            )
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @property
    def embedding_dim(self):
        return int(self.model.config.hidden_size)

    def encode(self, texts, batch_size=None, show_progress_bar=False, convert_to_numpy=True):
        del show_progress_bar
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        outputs = []
        effective_batch_size = batch_size or self.batch_size
        with torch.no_grad():
            for start in range(0, len(texts), effective_batch_size):
                batch_texts = texts[start:start + effective_batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                cls_vectors = self.model(**encoded).last_hidden_state[:, 0, :]
                outputs.append(cls_vectors.cpu())

        embeddings = torch.cat(outputs, dim=0)
        if convert_to_numpy:
            embeddings = embeddings.numpy().astype(np.float32)
            return embeddings[0] if single_input else embeddings
        return embeddings[0] if single_input else embeddings

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.tokenizer.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir)
