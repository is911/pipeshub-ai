from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx
import torch
from sentence_transformers import CrossEncoder

from app.models.blocks import BlockType, GroupType
from app.utils.aimodels import RerankerProvider
from app.utils.logger import create_logger

logger = create_logger("reranker")


class BaseRerankerService(ABC):
    """Abstract base class for reranker services"""

    @abstractmethod
    async def rerank(
        self, query: str, documents: List[Dict[str, Any]], top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank documents based on relevance to the query"""
        pass

    def _prepare_documents(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> tuple[List[tuple[str, str]], List[int]]:
        """Prepare document-query pairs for scoring, returning pairs and valid indices"""
        doc_query_pairs = []
        valid_indices = []

        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            if content:
                block_type = doc.get("block_type")
                if block_type == GroupType.TABLE.value:
                    text = content[0] if isinstance(content, list) and content else str(content)
                    doc_query_pairs.append((query, text))
                    valid_indices.append(i)
                elif block_type != BlockType.IMAGE.value:
                    text = content if isinstance(content, str) else str(content)
                    doc_query_pairs.append((query, text))
                    valid_indices.append(i)

        return doc_query_pairs, valid_indices

    def _apply_scores(
        self,
        documents: List[Dict[str, Any]],
        scores: List[float],
        valid_indices: List[int],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Apply reranker scores to documents and sort by final score"""
        score_map = {idx: score for idx, score in zip(valid_indices, scores)}

        for i, doc in enumerate(documents):
            if i in score_map:
                doc["reranker_score"] = float(score_map[i])
                if "score" in doc:
                    # Weighted combination: 30% retriever, 70% reranker
                    doc["final_score"] = 0.3 * doc["score"] + 0.7 * doc["reranker_score"]
                else:
                    doc["final_score"] = doc["reranker_score"]
            else:
                doc["reranker_score"] = 0.0
                doc["final_score"] = doc.get("score", 0.0)

        # Sort by final score descending
        reranked_docs = sorted(
            documents, key=lambda d: d.get("final_score", 0), reverse=True
        )

        if top_k is not None:
            return reranked_docs[:top_k]
        return reranked_docs

    def _set_default_scores(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Set default scores when reranking fails or no valid pairs"""
        for doc in documents:
            doc["reranker_score"] = 0.0
            doc["final_score"] = doc.get("score", 0.0)
        return documents


class LocalRerankerService(BaseRerankerService):
    """Local cross-encoder based reranker service"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        """
        Initialize with a local cross-encoder model

        Args:
            model_name: HuggingFace model name. Options:
                - "cross-encoder/ms-marco-MiniLM-L-6-v2" (fast)
                - "BAAI/bge-reranker-base" (balanced, default)
                - "BAAI/bge-reranker-large" (more accurate)
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=self.device)

        # Use half precision on GPU for faster inference
        if self.device == "cuda":
            self.model.model = self.model.model.half()

        logger.info(f"Initialized local reranker with model {model_name} on {self.device}")

    async def rerank(
        self, query: str, documents: List[Dict[str, Any]], top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        doc_query_pairs, valid_indices = self._prepare_documents(query, documents)

        if not doc_query_pairs:
            return self._set_default_scores(documents)

        try:
            scores = self.model.predict(doc_query_pairs)
            return self._apply_scores(documents, list(scores), valid_indices, top_k)
        except Exception as e:
            logger.error(f"Local reranker error: {e}")
            return self._set_default_scores(documents)


class VoyageRerankerService(BaseRerankerService):
    """Voyage AI reranker service"""

    def __init__(
        self,
        api_key: str,
        model_name: str = "rerank-2.5",
        base_url: str = "https://api.voyageai.com/v1/rerank",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        logger.info(f"Initialized Voyage reranker with model {model_name}")

    async def rerank(
        self, query: str, documents: List[Dict[str, Any]], top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        doc_query_pairs, valid_indices = self._prepare_documents(query, documents)

        if not doc_query_pairs:
            return self._set_default_scores(documents)

        # Extract just the document texts for the API
        doc_texts = [pair[1] for pair in doc_query_pairs]

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "query": query,
                        "documents": doc_texts,
                        "model": self.model_name,
                        "top_k": len(doc_texts),  # Get all scores, we'll filter later
                    },
                )
                response.raise_for_status()
                result = response.json()

            # Map API response indices back to original indices
            api_scores = {r["index"]: r["relevance_score"] for r in result.get("results", [])}
            scores = [api_scores.get(i, 0.0) for i in range(len(doc_texts))]

            return self._apply_scores(documents, scores, valid_indices, top_k)

        except Exception as e:
            logger.error(f"Voyage reranker error: {e}")
            return self._set_default_scores(documents)


class CohereRerankerService(BaseRerankerService):
    """Cohere reranker service"""

    def __init__(
        self,
        api_key: str,
        model_name: str = "rerank-v3.5",
        base_url: str = "https://api.cohere.com/v2/rerank",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        logger.info(f"Initialized Cohere reranker with model {model_name}")

    async def rerank(
        self, query: str, documents: List[Dict[str, Any]], top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        doc_query_pairs, valid_indices = self._prepare_documents(query, documents)

        if not doc_query_pairs:
            return self._set_default_scores(documents)

        doc_texts = [pair[1] for pair in doc_query_pairs]

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "query": query,
                        "documents": doc_texts,
                        "model": self.model_name,
                        "top_n": len(doc_texts),
                    },
                )
                response.raise_for_status()
                result = response.json()

            # Cohere returns results sorted by relevance, map back to original indices
            api_scores = {r["index"]: r["relevance_score"] for r in result.get("results", [])}
            scores = [api_scores.get(i, 0.0) for i in range(len(doc_texts))]

            return self._apply_scores(documents, scores, valid_indices, top_k)

        except Exception as e:
            logger.error(f"Cohere reranker error: {e}")
            return self._set_default_scores(documents)


class JinaRerankerService(BaseRerankerService):
    """Jina AI reranker service"""

    def __init__(
        self,
        api_key: str,
        model_name: str = "jina-reranker-v2-base-multilingual",
        base_url: str = "https://api.jina.ai/v1/rerank",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        logger.info(f"Initialized Jina reranker with model {model_name}")

    async def rerank(
        self, query: str, documents: List[Dict[str, Any]], top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        doc_query_pairs, valid_indices = self._prepare_documents(query, documents)

        if not doc_query_pairs:
            return self._set_default_scores(documents)

        doc_texts = [pair[1] for pair in doc_query_pairs]

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "query": query,
                        "documents": doc_texts,
                        "model": self.model_name,
                        "top_n": len(doc_texts),
                    },
                )
                response.raise_for_status()
                result = response.json()

            # Jina returns results with index and relevance_score
            api_scores = {r["index"]: r["relevance_score"] for r in result.get("results", [])}
            scores = [api_scores.get(i, 0.0) for i in range(len(doc_texts))]

            return self._apply_scores(documents, scores, valid_indices, top_k)

        except Exception as e:
            logger.error(f"Jina reranker error: {e}")
            return self._set_default_scores(documents)


def get_reranker_service(
    provider: str, config: Dict[str, Any]
) -> BaseRerankerService:
    """
    Factory function to create a reranker service based on provider

    Args:
        provider: Provider name (local, voyage, cohere, jinaAI)
        config: Provider configuration containing apiKey, model, etc.

    Returns:
        Configured reranker service instance
    """
    configuration = config.get("configuration", config)
    model_name = configuration.get("model", "")

    if provider == RerankerProvider.LOCAL.value:
        return LocalRerankerService(
            model_name=model_name or "BAAI/bge-reranker-base"
        )

    elif provider == RerankerProvider.VOYAGE.value:
        return VoyageRerankerService(
            api_key=configuration["apiKey"],
            model_name=model_name or "rerank-2.5",
        )

    elif provider == RerankerProvider.COHERE.value:
        return CohereRerankerService(
            api_key=configuration["apiKey"],
            model_name=model_name or "rerank-v3.5",
        )

    elif provider == RerankerProvider.JINA_AI.value:
        return JinaRerankerService(
            api_key=configuration["apiKey"],
            model_name=model_name or "jina-reranker-v2-base-multilingual",
        )

    else:
        raise ValueError(f"Unsupported reranker provider: {provider}")


# Legacy alias for backward compatibility
RerankerService = LocalRerankerService
