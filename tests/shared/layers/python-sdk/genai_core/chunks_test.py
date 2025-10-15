import importlib
import sys
import types
from unittest import mock

import pytest


class _DummyRecursiveCharacterTextSplitter:
    def __init__(self, *_, **__):
        pass

    def split_text(self, text):
        return [text]


def _build_workspace(strategy="semantic"):
    return {
        "chunking_strategy": strategy,
        "chunk_size": 256,
        "chunk_overlap": 32,
        "embeddings_model_provider": "bedrock",
        "embeddings_model_name": "test-model",
    }


@pytest.fixture()
def chunks_module(monkeypatch):
    class _S3Object:
        def put(self, *_, **__):
            return None

    class _S3Resource:
        def Object(self, *_, **__):
            return _S3Object()

    powertools_stub = types.ModuleType("aws_lambda_powertools")
    powertools_stub.Logger = lambda *_, **__: None
    powertools_utilities_stub = types.ModuleType("aws_lambda_powertools.utilities")
    powertools_parameters_stub = types.ModuleType(
        "aws_lambda_powertools.utilities.parameters"
    )
    powertools_parameters_stub.get_secret = lambda *_, **__: ""
    powertools_utilities_stub.parameters = powertools_parameters_stub
    powertools_stub.utilities = powertools_utilities_stub
    monkeypatch.setitem(sys.modules, "aws_lambda_powertools", powertools_stub)
    monkeypatch.setitem(
        sys.modules, "aws_lambda_powertools.utilities", powertools_utilities_stub
    )
    monkeypatch.setitem(
        sys.modules,
        "aws_lambda_powertools.utilities.parameters",
        powertools_parameters_stub,
    )

    botocore_stub = types.ModuleType("botocore")
    botocore_stub.exceptions = types.SimpleNamespace(ClientError=Exception)
    botocore_config_stub = types.ModuleType("botocore.config")
    botocore_config_stub.Config = object
    botocore_stub.config = botocore_config_stub
    monkeypatch.setitem(sys.modules, "botocore", botocore_stub)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_config_stub)

    class _BaseModel:
        def __init__(self, *_, **__):
            pass

    pydantic_stub = types.ModuleType("pydantic")
    pydantic_stub.BaseModel = _BaseModel
    monkeypatch.setitem(sys.modules, "pydantic", pydantic_stub)

    documents_stub = types.ModuleType("genai_core.documents")
    documents_stub.set_document_vectors = lambda *_, **__: None
    monkeypatch.setitem(sys.modules, "genai_core.documents", documents_stub)

    embeddings_stub = types.ModuleType("genai_core.embeddings")
    embeddings_stub.get_embeddings_model = lambda *_, **__: object()
    embeddings_stub.generate_embeddings = lambda *_, **__: []
    monkeypatch.setitem(sys.modules, "genai_core.embeddings", embeddings_stub)

    aurora_chunks_stub = types.ModuleType("genai_core.aurora.chunks")
    aurora_chunks_stub.add_chunks_aurora = (
        lambda *_, **__: {"added_vectors": 0}
    )
    aurora_pkg_stub = types.ModuleType("genai_core.aurora")
    aurora_pkg_stub.__path__ = []
    aurora_pkg_stub.chunks = aurora_chunks_stub
    monkeypatch.setitem(sys.modules, "genai_core.aurora", aurora_pkg_stub)
    monkeypatch.setitem(sys.modules, "genai_core.aurora.chunks", aurora_chunks_stub)

    opensearch_chunks_stub = types.ModuleType("genai_core.opensearch.chunks")
    opensearch_chunks_stub.add_chunks_open_search = (
        lambda *_, **__: {"added_vectors": 0}
    )
    opensearch_pkg_stub = types.ModuleType("genai_core.opensearch")
    opensearch_pkg_stub.__path__ = []
    opensearch_pkg_stub.chunks = opensearch_chunks_stub
    monkeypatch.setitem(sys.modules, "genai_core.opensearch", opensearch_pkg_stub)
    monkeypatch.setitem(
        sys.modules, "genai_core.opensearch.chunks", opensearch_chunks_stub
    )

    boto3_stub = types.ModuleType("boto3")
    boto3_stub.resource = lambda *_, **__: _S3Resource()
    boto3_stub.client = lambda *_, **__: types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "boto3", boto3_stub)

    splitters_stub = types.ModuleType("langchain_text_splitters")
    splitters_stub.RecursiveCharacterTextSplitter = _DummyRecursiveCharacterTextSplitter
    splitters_stub.SemanticChunker = object
    monkeypatch.setitem(sys.modules, "langchain_text_splitters", splitters_stub)

    langchain_core_stub = types.ModuleType("langchain_core")
    langchain_core_stub.__path__ = []
    langchain_embeddings_stub = types.ModuleType("langchain_core.embeddings")
    langchain_embeddings_stub.Embeddings = object
    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core_stub)
    monkeypatch.setitem(sys.modules, "langchain_core.embeddings", langchain_embeddings_stub)

    sys.modules.pop("genai_core.chunks", None)
    module = importlib.import_module("genai_core.chunks")

    documents_stub = types.SimpleNamespace(set_document_vectors=lambda *_, **__: None)
    embeddings_stub = types.SimpleNamespace(
        get_embeddings_model=lambda *_, **__: object(),
        generate_embeddings=lambda *_, **__: [],
    )
    aurora_stub = types.SimpleNamespace(
        chunks=types.SimpleNamespace(add_chunks_aurora=lambda *_, **__: {"added_vectors": 0})
    )
    opensearch_stub = types.SimpleNamespace(
        chunks=types.SimpleNamespace(
            add_chunks_open_search=lambda *_, **__: {"added_vectors": 0}
        )
    )

    monkeypatch.setattr(module.genai_core, "documents", documents_stub, raising=False)
    monkeypatch.setattr(module.genai_core, "embeddings", embeddings_stub, raising=False)
    monkeypatch.setattr(module.genai_core, "aurora", aurora_stub, raising=False)
    monkeypatch.setattr(module.genai_core, "opensearch", opensearch_stub, raising=False)

    yield module

    sys.modules.pop("genai_core.chunks", None)


def test_split_content_requires_semantic_dependencies(chunks_module):
    with mock.patch.object(chunks_module, "SemanticChunker", None), mock.patch.object(
        chunks_module, "Embeddings", None
    ):
        with pytest.raises(chunks_module.NewCommonError) as exc_info:
            chunks_module.split_content(_build_workspace(), "some content")

    assert "dependencies" in str(exc_info.value)


def test_split_content_requires_embeddings_model(chunks_module):
    with mock.patch.object(
        chunks_module.genai_core.embeddings, "get_embeddings_model", return_value=None
    ):
        with pytest.raises(chunks_module.NewCommonError) as exc_info:
            chunks_module.split_content(_build_workspace(), "content")

    assert "Embeddings model not found" in str(exc_info.value)


def test_split_content_raises_when_semantic_chunking_fails(chunks_module):
    mock_embeddings_model = object()

    with mock.patch.object(
        chunks_module.genai_core.embeddings,
        "get_embeddings_model",
        return_value=mock_embeddings_model,
    ), mock.patch("genai_core.chunks.SemanticChunker") as semantic_cls:
        semantic_instance = semantic_cls.return_value
        semantic_instance.split_text.side_effect = RuntimeError("boom")
        with pytest.raises(chunks_module.NewCommonError) as exc_info:
            chunks_module.split_content(_build_workspace(), "content")

    assert "Semantic chunking failed" in str(exc_info.value)
