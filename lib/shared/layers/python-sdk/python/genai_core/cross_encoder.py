import os
import json
import genai_core.types
import genai_core.clients
import genai_core.parameters
from typing import List, Optional


SAGEMAKER_RAG_MODELS_ENDPOINT = os.environ.get("SAGEMAKER_RAG_MODELS_ENDPOINT")


def rank_passages(
    model: genai_core.types.CrossEncoderModel, input: str, passages: List[str]
):
    input = input[:10000]
    passages = passages[:1000]
    passages = list(map(lambda x: x[:10000], passages))

    if model.provider == "sagemaker":
        return _rank_passages_sagemaker(model, input, passages)
    elif model.provider == "bedrock":
        return _rank_passages_bedrock(model, input, passages)

    raise genai_core.types.CommonError("Unknown provider")


def get_cross_encoder_models():
    config = genai_core.parameters.get_config()
    models = config["rag"]["crossEncoderModels"]

    if not SAGEMAKER_RAG_MODELS_ENDPOINT:
        models = list(filter(lambda x: x["provider"] != "sagemaker", models))

    return models


def get_cross_encoder_model(
    provider: str, name: str
) -> Optional[genai_core.types.CrossEncoderModel]:
    config = genai_core.parameters.get_config()
    models = config["rag"]["crossEncoderModels"]

    for model in models:
        if model["provider"] == provider and model["name"] == name:
            return genai_core.types.CrossEncoderModel(**model)

    return None


def _rank_passages_sagemaker(
    model: genai_core.types.CrossEncoderModel, input: str, passages: List[str]
):
    client = genai_core.clients.get_sagemaker_client()

    response = client.invoke_endpoint(
        EndpointName=SAGEMAKER_RAG_MODELS_ENDPOINT,
        ContentType="application/json",
        Body=json.dumps(
            {
                "type": "cross-encoder",
                "model": model.name,
                "input": input,
                "passages": passages,
            }
        ),
    )

    ret_value = json.loads(response["Body"].read().decode())

    return ret_value


def _rank_passages_bedrock(
    model: genai_core.types.CrossEncoderModel, input: str, passages: List[str]
):
    client = genai_core.clients.get_bedrock_client()

    if not client:
        raise genai_core.types.CommonError("Bedrock is not enabled.")

    # Extract the model provider from the model name
    model_provider = model.name.split(".")[0]
    
    if model_provider == "cohere":
        # Cohere rerank-v3-5 format
        body_data = {
            "query": input,
            "documents": passages,
            "top_n": len(passages),
            "api_version": 2  # Required by Cohere's rerank model
        }
    elif model_provider == "amazon":
        # Amazon rerank-v1 format
        body_data = {
            "query": input,
            "documents": passages,
            "top_n": len(passages)
            # No api_version needed for Amazon's model
        }
    else:
        raise genai_core.types.CommonError(f'Unknown reranking provider "{model_provider}"')

    body = json.dumps(body_data)
    
    response = client.invoke_model(
        modelId=model.name,
        body=body,
        contentType="application/json",
        accept="*/*"  # Using wildcard accept as shown in the examples
    )

    response_body = json.loads(response.get('body').read())
    
    if model_provider == "cohere":
        # Cohere returns a list of results with index and relevance_score
        # Example: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
        return [result.get("relevance_score", 0.0) for result in response_body.get("results", [])]
    elif model_provider == "amazon":
        # Amazon returns a list of scores in the same order as input documents
        # Example: {"scores": [0.9, 0.3, ...]}
        return response_body.get("scores", [0.0] * len(passages))
    else:
        raise genai_core.types.CommonError(f'Unknown reranking provider "{model_provider}"')
