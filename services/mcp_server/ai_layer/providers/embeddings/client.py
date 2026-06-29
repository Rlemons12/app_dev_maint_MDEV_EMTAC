from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ai_layer.providers.embeddings.settings import (
    EmbeddingSettings,
    get_embedding_settings,
)


HF_INFERENCE_PROVIDERS = {"hf-inference", "huggingface", "huggingface-inference"}
LOCAL_SENTENCE_TRANSFORMERS_PROVIDERS = {"sentence-transformers", "local-huggingface"}
SUPPORTED_EMBEDDING_PROVIDERS = (
    HF_INFERENCE_PROVIDERS
    | LOCAL_SENTENCE_TRANSFORMERS_PROVIDERS
    | {"ollama", "openai"}
)


class EmbeddingClientError(RuntimeError):
    """Raised when an embedding provider cannot create embeddings."""


@dataclass(frozen=True)
class EmbeddingResult:
    provider: str
    model: str
    embeddings: list[list[float]]
    dimensions: int
    usage: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "count": len(self.embeddings),
            "dimensions": self.dimensions,
            "embeddings": self.embeddings,
        }

        if self.usage is not None:
            result["usage"] = self.usage

        return result


class EmbeddingClient:
    def __init__(self, settings: EmbeddingSettings | None = None) -> None:
        self.settings = settings or get_embedding_settings()

    def configured_provider(self) -> dict[str, Any]:
        return {
            "provider": self.settings.provider,
            "model": self.settings.model,
            "dimensions": self.settings.dimensions,
            "timeout_seconds": self.settings.timeout_seconds,
            "supported_providers": sorted(SUPPORTED_EMBEDDING_PROVIDERS),
            "openai_api_key_set": bool(self.settings.openai_api_key),
            "ollama_base_url": self.settings.ollama_base_url,
            "huggingface_cache_dir": self.settings.huggingface_cache_dir,
            "huggingface_token_set": bool(self.settings.huggingface_token),
            "huggingface_inference_provider": (
                self.settings.huggingface_inference_provider
            ),
        }

    def create_embeddings(
        self,
        texts: list[str],
        provider: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> EmbeddingResult:
        clean_texts = [text for text in texts if text.strip()]

        if not clean_texts:
            raise EmbeddingClientError("At least one non-empty text value is required.")

        selected_provider = (provider or self.settings.provider).lower().strip()
        selected_model = (model or self.settings.model).strip()
        selected_dimensions = dimensions or self.settings.dimensions

        if selected_provider not in SUPPORTED_EMBEDDING_PROVIDERS:
            supported = ", ".join(sorted(SUPPORTED_EMBEDDING_PROVIDERS))
            raise EmbeddingClientError(
                f"Unsupported embedding provider '{selected_provider}'. "
                f"Supported providers: {supported}."
            )

        if not selected_model:
            raise EmbeddingClientError("Embedding model is required.")

        if selected_provider == "openai":
            return self._create_openai_embeddings(
                clean_texts,
                selected_model,
                selected_dimensions,
            )

        if selected_provider == "ollama":
            return self._create_ollama_embeddings(clean_texts, selected_model)

        if selected_provider in HF_INFERENCE_PROVIDERS:
            return self.create_huggingface_feature_extraction(
                texts=clean_texts,
                model=selected_model,
            )

        return self._create_sentence_transformers_embeddings(clean_texts, selected_model)

    def create_huggingface_feature_extraction(
        self,
        texts: list[str],
        model: str | None = None,
        normalize: bool | None = None,
        truncate: bool | None = None,
        prompt_name: str | None = None,
        truncation_direction: str | None = None,
        dimensions: int | None = None,
    ) -> EmbeddingResult:
        clean_texts = [text for text in texts if text.strip()]

        if not clean_texts:
            raise EmbeddingClientError("At least one non-empty text value is required.")

        if model:
            selected_model = model.strip()
        elif self.settings.provider in HF_INFERENCE_PROVIDERS:
            selected_model = self.settings.model.strip()
        else:
            selected_model = "sentence-transformers/all-MiniLM-L6-v2"

        if not selected_model:
            raise EmbeddingClientError("Hugging Face inference model is required.")

        if not self.settings.huggingface_token:
            raise EmbeddingClientError(
                "HF_TOKEN is required for Hugging Face hosted inference."
            )

        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise EmbeddingClientError(
                "The huggingface_hub package is required for Hugging Face inference. "
                "Install requirements.txt or add huggingface_hub."
            ) from exc

        client = InferenceClient(
            provider=self.settings.huggingface_inference_provider,
            api_key=self.settings.huggingface_token,
            timeout=self.settings.timeout_seconds,
        )
        request_kwargs: dict[str, Any] = {"model": selected_model}

        if normalize is not None:
            request_kwargs["normalize"] = normalize

        if truncate is not None:
            request_kwargs["truncate"] = truncate

        if prompt_name:
            request_kwargs["prompt_name"] = prompt_name

        if truncation_direction:
            request_kwargs["truncation_direction"] = truncation_direction

        if dimensions is not None:
            request_kwargs["dimensions"] = dimensions

        result = client.feature_extraction(clean_texts, **request_kwargs)
        embeddings = _coerce_embeddings(result)

        return EmbeddingResult(
            provider="hf-inference",
            model=selected_model,
            embeddings=embeddings,
            dimensions=_embedding_dimensions(embeddings),
        )

    def _create_openai_embeddings(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None,
    ) -> EmbeddingResult:
        if not self.settings.openai_api_key:
            raise EmbeddingClientError("OPENAI_API_KEY is required for OpenAI embeddings.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise EmbeddingClientError(
                "The openai package is required for OpenAI embeddings. "
                "Install requirements.txt or add openai to the environment."
            ) from exc

        client_kwargs: dict[str, Any] = {"api_key": self.settings.openai_api_key}

        if self.settings.openai_base_url:
            client_kwargs["base_url"] = self.settings.openai_base_url

        client = OpenAI(**client_kwargs, timeout=self.settings.timeout_seconds)
        request_kwargs: dict[str, Any] = {"model": model, "input": texts}

        if dimensions is not None:
            request_kwargs["dimensions"] = dimensions

        response = client.embeddings.create(**request_kwargs)
        embeddings = [item.embedding for item in response.data]

        return EmbeddingResult(
            provider="openai",
            model=model,
            embeddings=embeddings,
            dimensions=_embedding_dimensions(embeddings),
            usage=_json_safe(response.usage) if response.usage else None,
        )

    def _create_ollama_embeddings(
        self,
        texts: list[str],
        model: str,
    ) -> EmbeddingResult:
        endpoint = self.settings.ollama_base_url.rstrip("/") + "/api/embed"
        body = json.dumps({"model": model, "input": texts}).encode("utf-8")
        request = Request(
            endpoint,
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise EmbeddingClientError(
                f"Ollama returned HTTP {exc.code}: {exc.reason}"
            ) from exc
        except URLError as exc:
            raise EmbeddingClientError(
                f"Could not connect to Ollama at {self.settings.ollama_base_url}: "
                f"{exc.reason}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise EmbeddingClientError("Ollama returned invalid JSON.") from exc

        embeddings = payload.get("embeddings")

        if not isinstance(embeddings, list):
            raise EmbeddingClientError("Ollama response did not include embeddings.")

        return EmbeddingResult(
            provider="ollama",
            model=model,
            embeddings=embeddings,
            dimensions=_embedding_dimensions(embeddings),
        )

    def _create_sentence_transformers_embeddings(
        self,
        texts: list[str],
        model: str,
    ) -> EmbeddingResult:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingClientError(
                "The sentence-transformers provider requires sentence-transformers, "
                "transformers, torch, and safetensors in this Python environment."
            ) from exc

        model_path = _resolve_huggingface_model_path(
            model,
            Path(self.settings.huggingface_cache_dir),
        )
        encoder = SentenceTransformer(str(model_path))
        embeddings = encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        return EmbeddingResult(
            provider="sentence-transformers",
            model=model,
            embeddings=embeddings.tolist(),
            dimensions=int(embeddings.shape[1]) if len(embeddings.shape) > 1 else 0,
        )


def _embedding_dimensions(embeddings: list[list[float]]) -> int:
    if not embeddings:
        return 0

    return len(embeddings[0])


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())

    return str(value)


def _coerce_embeddings(value: Any) -> list[list[float]]:
    if hasattr(value, "tolist"):
        value = value.tolist()

    if not isinstance(value, list):
        raise EmbeddingClientError(
            "Hugging Face inference returned an unexpected embedding value."
        )

    if not value:
        return []

    if all(isinstance(item, (int, float)) for item in value):
        return [[float(item) for item in value]]

    embeddings: list[list[float]] = []

    for embedding in value:
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        if not isinstance(embedding, list):
            raise EmbeddingClientError(
                "Hugging Face inference returned a non-list embedding."
            )

        embeddings.append([float(item) for item in embedding])

    return embeddings


def _resolve_huggingface_model_path(model: str, cache_dir: Path) -> Path:
    model_path = Path(model)

    if model_path.exists():
        return model_path

    cache_model_dir = cache_dir / ("models--" + model.replace("/", "--"))

    if not cache_model_dir.exists():
        raise EmbeddingClientError(
            f"Hugging Face model '{model}' is not cached at {cache_model_dir}."
        )

    ref_file = cache_model_dir / "refs" / "main"

    if ref_file.exists():
        snapshot_id = ref_file.read_text(encoding="utf-8").strip()
        snapshot_dir = cache_model_dir / "snapshots" / snapshot_id

        if snapshot_dir.exists():
            return snapshot_dir

    snapshots_dir = cache_model_dir / "snapshots"

    if snapshots_dir.exists():
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]

        if snapshots:
            return max(snapshots, key=lambda path: path.stat().st_mtime)

    raise EmbeddingClientError(
        f"Hugging Face model '{model}' does not have a usable cached snapshot."
    )
