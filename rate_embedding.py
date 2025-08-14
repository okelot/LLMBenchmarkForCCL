import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util


class EmbeddingEvaluator:
    """
    Longer-text ready semantic similarity:
      - Token-based chunking with overlap to avoid truncation
      - Batch encoding for speed
      - L2-normalized embeddings and mean pooling across chunks
    """
    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",       # better for longer text than MiniLM
        device: Optional[str] = None,                # "cuda", "cpu", or None (auto)
        normalize_embeddings: bool = True,
        chunk_overlap: int = 32,                     # token overlap between chunks
        max_tokens: Optional[int] = None,            # if None, use model's max_seq_length
        batch_size: int = 32
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.tokenizer = self.model.tokenizer
        self.normalize = normalize_embeddings
        self.batch_size = batch_size

        # Determine usable token window (reserve small margin for specials)
        model_max = getattr(self.model, "max_seq_length", None) or getattr(self.model, "get_max_seq_length", lambda: 512)()
        if callable(model_max):
            model_max = model_max()
        self.max_tokens = max_tokens or int(model_max)
        self.margin = 8  # safety margin for special tokens
        self.chunk_size = max(16, self.max_tokens - self.margin)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))

    # ---------- chunking ----------
    def _chunk_by_tokens(self, text: str) -> List[str]:
        """Split text into overlapping chunks of ~chunk_size tokens using the model tokenizer."""
        if not isinstance(text, str) or not text.strip():
            return []

        # Encode without adding specials; we'll decode chunks back to text
        ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False
        )
        if len(ids) <= self.chunk_size:
            return [text]

        chunks = []
        stride = self.chunk_size - self.chunk_overlap
        for start in range(0, len(ids), stride):
            end = start + self.chunk_size
            sub_ids = ids[start:end]
            if not sub_ids:
                break
            # Decode chunk; skip_special_tokens not needed since we excluded specials
            chunk = self.tokenizer.decode(sub_ids, skip_special_tokens=True)
            chunks.append(chunk)
            if end >= len(ids):
                break
        return chunks

    # ---------- embedding & pooling ----------
    def _embed_texts(self, texts: List[str]) -> torch.Tensor:
        """Embed a list of texts in batches; returns (N, D) tensor."""
        if not texts:
            return torch.empty((0, self.model.get_sentence_embedding_dimension()))
        # encode supports batching internally, but we control batch_size for stability
        embs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        return embs

    def _pooled_embedding(self, text: str) -> Optional[torch.Tensor]:
        """Compute a single pooled embedding for possibly-long text via chunking + mean pool."""
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            return None

        chunks = self._chunk_by_tokens(text)
        if not chunks:
            return None
        embs = self._embed_texts(chunks)  # (K, D)
        if embs.shape[0] == 0:
            return None
        # Mean-pool across chunk embeddings, then (optionally) normalize again
        pooled = embs.mean(dim=0, keepdim=True)  # (1, D)
        if self.normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.squeeze(0)  # (D,)

    # ---------- public API ----------
    def compute_similarity(self, text1: str, text2: str) -> Optional[float]:
        """
        Cosine similarity between two (possibly long) texts:
          - chunk -> embed -> mean-pool -> cosine
        Returns None if either text is missing/empty.
        """
        e1 = self._pooled_embedding(text1)
        e2 = self._pooled_embedding(text2)
        if e1 is None or e2 is None:
            return None
        # util.cos_sim expects 2D tensors; add batch axis
        sim = util.cos_sim(e1.unsqueeze(0), e2.unsqueeze(0)).item()
        return float(sim)

    def evaluate_results(self, input_file: str) -> str:
        """Evaluate section-wise similarity from a CSV file."""
        df = pd.read_csv(input_file)

        required = [
            "ai_facts", "human_facts",
            "ai_issue", "human_issue",
            "ai_decision", "human_decision",
            "ai_reasons", "human_reasons",
            "ai_ratio", "human_ratio",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        sections = ["facts", "issue", "decision", "reasons", "ratio"]
        for section in sections:
            col_name = f"{section}_similarity"
            df[col_name] = [
                self.compute_similarity(row[f"ai_{section}"], row[f"human_{section}"])
                for _, row in df.iterrows()
            ]
            print(f"Finished computing {col_name}")

        # Save results
        Path("results").mkdir(exist_ok=True)
        output_file = "results/evaluated_case_model_results_with_section_similarity.csv"
        df.to_csv(output_file, index=False)
        print(f"Evaluation complete. Results saved to {output_file}")
        return output_file


def main(input_file: str):
    evaluator = EmbeddingEvaluator(
        model_name="all-mpnet-base-v2",  # good quality, 512-token window
        device=None,                     # auto: "cuda" if available else "cpu"
        normalize_embeddings=True,
        chunk_overlap=32,
        max_tokens=None,                 # use model default
        batch_size=32
    )
    return evaluator.evaluate_results(input_file)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python rate_llm.py <input_csv_file>")
        sys.exit(1)
    main(sys.argv[1])
