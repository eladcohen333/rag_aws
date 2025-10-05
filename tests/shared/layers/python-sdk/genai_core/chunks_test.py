import importlib
import sys
import types
from unittest import mock

import pytest


mock_boto3 = types.ModuleType("boto3")
mock_boto3.resource = lambda *_, **__: None
sys.modules.setdefault("boto3", mock_boto3)


class _DummyRecursiveCharacterTextSplitter:
    def __init__(self, *_, **__):
        pass

    def split_text(self, text):
        return [text]


langchain_text_splitters = types.ModuleType("langchain_text_splitters")
langchain_text_splitters.RecursiveCharacterTextSplitter = (
    _DummyRecursiveCharacterTextSplitter
)
langchain_text_splitters.SemanticChunker = object
sys.modules.setdefault("langchain_text_splitters", langchain_text_splitters)

langchain_core_embeddings = types.ModuleType("langchain_core.embeddings")
langchain_core_embeddings.Embeddings = object
sys.modules.setdefault("langchain_core.embeddings", langchain_core_embeddings)

aws_lambda_powertools = types.ModuleType("aws_lambda_powertools")
aws_lambda_powertools.Logger = lambda *_, **__: None
sys.modules.setdefault("aws_lambda_powertools", aws_lambda_powertools)

genai_core_documents = types.ModuleType("genai_core.documents")
genai_core_documents.set_document_vectors = lambda *_, **__: None
sys.modules.setdefault("genai_core.documents", genai_core_documents)

genai_core_embeddings = types.ModuleType("genai_core.embeddings")
genai_core_embeddings.get_embeddings_model = lambda *_, **__: None
genai_core_embeddings.generate_embeddings = lambda *_, **__: []
sys.modules.setdefault("genai_core.embeddings", genai_core_embeddings)


genai_core_aurora = types.ModuleType("genai_core.aurora")
genai_core_aurora.chunks = types.SimpleNamespace(
    add_chunks_aurora=lambda *_, **__: {"added_vectors": 0}
)
sys.modules.setdefault("genai_core.aurora", genai_core_aurora)

genai_core_aurora_chunks = types.ModuleType("genai_core.aurora.chunks")
genai_core_aurora_chunks.add_chunks_aurora = (
    lambda *_, **__: {"added_vectors": 0}
)
sys.modules.setdefault("genai_core.aurora.chunks", genai_core_aurora_chunks)

genai_core_opensearch = types.ModuleType("genai_core.opensearch")
genai_core_opensearch.chunks = types.SimpleNamespace(
    add_chunks_open_search=lambda *_, **__: {"added_vectors": 0}
)
sys.modules.setdefault("genai_core.opensearch", genai_core_opensearch)

genai_core_opensearch_chunks = types.ModuleType("genai_core.opensearch.chunks")
genai_core_opensearch_chunks.add_chunks_open_search = (
    lambda *_, **__: {"added_vectors": 0}
)
sys.modules.setdefault(
    "genai_core.opensearch.chunks", genai_core_opensearch_chunks
)

genai_core_types = types.ModuleType("genai_core.types")
genai_core_types.CommonError = Exception
genai_core_types.Task = types.SimpleNamespace(
    STORE=types.SimpleNamespace(value="store"),
    RETRIEVE=types.SimpleNamespace(value="retrieve"),
)
genai_core_types.EmbeddingsModel = None
genai_core_types.Provider = None
sys.modules.setdefault("genai_core.types", genai_core_types)

chunks_module = importlib.import_module("genai_core.chunks")
setattr(sys.modules["genai_core"], "embeddings", genai_core_embeddings)


def _build_workspace(strategy="semantic"):
    return {
        "chunking_strategy": strategy,
        "chunk_size": 256,
        "chunk_overlap": 32,
        "embeddings_model_provider": "bedrock",
        "embeddings_model_name": "test-model",
    }


def test_split_content_requires_semantic_dependencies():
    with mock.patch.object(chunks_module, "SemanticChunker", None), mock.patch.object(
        chunks_module, "Embeddings", None
    ):
        with pytest.raises(chunks_module.NewCommonError) as exc_info:
            chunks_module.split_content(_build_workspace(), "some content")

    assert "dependencies" in str(exc_info.value)


def test_split_content_requires_embeddings_model():
    with mock.patch.object(
        chunks_module.genai_core.embeddings, "get_embeddings_model", return_value=None
    ):
        with pytest.raises(chunks_module.NewCommonError) as exc_info:
            chunks_module.split_content(_build_workspace(), "content")

    assert "Embeddings model not found" in str(exc_info.value)


def test_split_content_raises_when_semantic_chunking_fails():
    mock_embeddings_model = object()

    with mock.patch.object(
        chunks_module.genai_core.embeddings, "get_embeddings_model", return_value=mock_embeddings_model
    ), mock.patch("genai_core.chunks.SemanticChunker") as semantic_cls:
        semantic_instance = semantic_cls.return_value
        semantic_instance.split_text.side_effect = RuntimeError("boom")
        with pytest.raises(chunks_module.NewCommonError) as exc_info:
            chunks_module.split_content(_build_workspace(), "content")

    assert "Semantic chunking failed" in str(exc_info.value)
