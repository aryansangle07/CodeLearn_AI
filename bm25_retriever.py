import os
import pickle
import re
from typing import List, Optional, Tuple
from rank_bm25 import BM25Okapi


class BM25Indexer:
    TOKEN_REGEX = re.compile(r"[a-zA-Z0-9_]+|[^\s\w]")
    CAMEL_SPLIT_REGEX = re.compile(r"([a-z])([A-Z])")

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_size: int = 0

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """
        Code-aware tokenizer splitting camelCase, snake_case, and punctuation tokens.
        Preserves full identifier tokens alongside decomposed sub-tokens.
        """
        if not text:
            return []

        raw_tokens = cls.TOKEN_REGEX.findall(text)
        token_list: List[str] = []

        for token in raw_tokens:
            token_lower = token.lower()
            token_list.append(token_lower)

            # Split snake_case
            if "_" in token:
                subparts = [p.lower() for p in token.split("_") if p]
                token_list.extend(subparts)

            # Split camelCase
            camel_split = cls.CAMEL_SPLIT_REGEX.sub(r"\1 \2", token)
            if " " in camel_split:
                token_list.extend([w.lower() for w in camel_split.split() if w])

        return token_list

    def fit(self, corpus_texts: List[str]) -> None:
        """
        Tokenizes the input text corpus and builds the BM25Okapi sparse lexical index.
        """
        if not corpus_texts:
            self.bm25 = None
            self.corpus_size = 0
            return

        tokenized_corpus = [self.tokenize(doc) for doc in corpus_texts]
        # Guard against completely empty token lists
        tokenized_corpus = [doc if doc else ["<empty>"] for doc in tokenized_corpus]

        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus_size = len(corpus_texts)

    def save(self, file_path: str) -> None:
        """Serializes the fitted BM25 model to disk via pickle."""
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        payload = {
            "bm25": self.bm25,
            "corpus_size": self.corpus_size,
        }
        with open(file_path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, file_path: str) -> None:
        """Loads a serialized BM25 index from a pickle file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"BM25 index file not found at: {file_path}")

        with open(file_path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.corpus_size = data["corpus_size"]

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Performs sparse BM25 scoring for a query string.
        Returns a list of (doc_id, bm25_score) tuples sorted descending by score.
        """
        if self.bm25 is None or self.corpus_size == 0:
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        # Pair with 0-indexed doc IDs
        scored_pairs = [(idx, float(score)) for idx, score in enumerate(scores)]
        # Sort descending by score
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        return scored_pairs[:top_k]
