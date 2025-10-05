import os
import json
import uuid
from datetime import datetime
from genai_core.registry import registry
from aws_lambda_powertools import Logger, Tracer
from aws_lambda_powertools.utilities import parameters
from aws_lambda_powertools.utilities.batch import BatchProcessor, EventType
from aws_lambda_powertools.utilities.batch.exceptions import BatchProcessingError
from aws_lambda_powertools.utilities.data_classes.sqs_event import SQSRecord
from aws_lambda_powertools.utilities.typing import LambdaContext

import adapters  # noqa: F401 Needed to register the adapters
from genai_core.utils.websocket import send_to_client
from genai_core.types import ChatbotAction

processor = BatchProcessor(event_type=EventType.SQS)
tracer = Tracer()
logger = Logger()

AWS_REGION = os.environ["AWS_REGION"]
API_KEYS_SECRETS_ARN = os.environ["API_KEYS_SECRETS_ARN"]

sequence_number = 0


def on_llm_new_token(
    user_id, session_id, self, token, run_id, chunk, parent_run_id, *args, **kwargs
):
    if self.disable_streaming:
        logger.debug("Streaming is disabled, ignoring token")
        return
    if isinstance(token, list):
        # When using the newer Chat objects from Langchain.
        # Token is not a string
        text = ""
        for t in token:
            if "text" in t:
                text = text + t.get("text")
    else:
        text = token
    if text is None or len(text) == 0:
        return
    global sequence_number
    sequence_number += 1
    run_id = str(run_id)

    send_to_client(
        {
            "type": "text",
            "action": ChatbotAction.LLM_NEW_TOKEN.value,
            "userId": user_id,
            "timestamp": str(int(round(datetime.now().timestamp()))),
            "data": {
                "sessionId": session_id,
                "token": {
                    "runId": run_id,
                    "sequenceNumber": sequence_number,
                    "value": text,
                },
            },
        }
    )


def handle_heartbeat(record):
    user_id = record["userId"]
    session_id = record["data"]["sessionId"]

    send_to_client(
        {
            "type": "text",
            "action": ChatbotAction.HEARTBEAT.value,
            "timestamp": str(int(round(datetime.now().timestamp()))),
            "userId": user_id,
            "data": {
                "sessionId": session_id,
            },
        }
    )


def preprocess_question(question):
    """
    Preprocess the question by removing punctuation and extra spaces
    """
    import re
    import string
    
    # Remove punctuation marks
    question = question.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra spaces (multiple spaces, leading/trailing spaces)
    question = re.sub(r'\s+', ' ', question).strip()
    
    return question


def generate_search_variations(question):
    """
    Generate different variations of the question for enhanced search
    """
    import re
    
    variations = []
    
    # Original question
    variations.append(question)
    
    # Remove common question words
    question_words = ['מה', 'איך', 'מתי', 'איפה', 'למה', 'מי', 'what', 'how', 'when', 'where', 'why', 'who']
    no_question_words = question
    for word in question_words:
        no_question_words = re.sub(r'\b' + word + r'\b', '', no_question_words, flags=re.IGNORECASE)
    no_question_words = re.sub(r'\s+', ' ', no_question_words).strip()
    if no_question_words and no_question_words != question:
        variations.append(no_question_words)
    
    # Extract key terms (words longer than 2 characters)
    key_terms = [word for word in question.split() if len(word) > 2]
    if len(key_terms) > 1:
        variations.append(' '.join(key_terms))
    
    # Try with only the most important words (longest words)
    if len(key_terms) > 2:
        sorted_terms = sorted(key_terms, key=len, reverse=True)
        variations.append(' '.join(sorted_terms[:3]))
    
    # Remove duplicates while preserving order
    unique_variations = []
    for var in variations:
        if var and var not in unique_variations:
            unique_variations.append(var)
    
    return unique_variations


def is_not_found_response(content):
    """
    Check if the response indicates that information was not found in the document
    """
    if not content:
        return False
    
    content_lower = content.lower()
    not_found_indicators = [
        "לא מצאתי במסמך",
        "לא נמצא במסמך",
        "אין מידע במסמך",
        "לא קיים במסמך",
        "המידע לא נמצא",
        "לא נמצא מידע",
        "אין תוכן רלוונטי",
        "לא נמצא תוכן",
        "לא יכול למצוא",
        "אין מידע זמין",
        "not found in document",
        "no information found",
        "cannot find",
        "no relevant information",
        "information not available",
        "i don't have information",
        "no data available",
        "unable to find"
    ]
    
    return any(indicator in content_lower for indicator in not_found_indicators)


def is_meaningful_response(content):
    """
    Check if the response contains meaningful information (not just a generic response)
    """
    if not content or len(content.strip()) < 10:
        return False
    
    # Check for generic responses that indicate no real information
    generic_responses = [
        "אני לא יודע",
        "לא יכול לענות",
        "אין לי מידע",
        "i don't know",
        "i cannot answer",
        "i'm not sure",
        "sorry, i don't have"
    ]
    
    content_lower = content.lower()
    has_generic = any(generic in content_lower for generic in generic_responses)
    
    # If it's not a generic response and has reasonable length, consider it meaningful
    return not has_generic and len(content.strip()) > 20


def handle_run(record):
    user_id = record["userId"]
    user_groups = record["userGroups"]
    data = record["data"]
    provider = data["provider"]
    model_id = data["modelName"]
    mode = data["mode"]
    original_prompt = data["text"]
    workspace_id = data.get("workspaceId", None)
    session_id = data.get("sessionId")
    images = data.get("images", [])
    documents = data.get("documents", [])
    videos = data.get("videos", [])
    system_prompts = record.get("systemPrompts", {})

    if not session_id:
        session_id = str(uuid.uuid4())

    adapter = registry.get_adapter(f"{provider}.{model_id}")

    adapter.on_llm_new_token = lambda *args, **kwargs: on_llm_new_token(
        user_id, session_id, *args, **kwargs
    )

    model = adapter(
        model_id=model_id,
        mode=mode,
        session_id=session_id,
        user_id=user_id,
        model_kwargs=data.get("modelKwargs", {}),
    )

    # Enhanced search strategy with multiple attempts
    best_response = None
    best_content = ""
    best_metadata = {}
    
    # Generate search variations for comprehensive search
    search_variations = []
    
    # Add preprocessed question
    processed_prompt = preprocess_question(original_prompt)
    search_variations.append(("processed", processed_prompt))
    
    # Add original question
    search_variations.append(("original", original_prompt))
    
    # Add question variations only if we have a workspace (RAG enabled)
    if workspace_id:
        variations = generate_search_variations(original_prompt)
        for i, variation in enumerate(variations[2:], 1):  # Skip first two (already added)
            search_variations.append((f"variation_{i}", variation))
    
    logger.info(f"Starting enhanced search with {len(search_variations)} variations")
    
    # Try each search variation until we get a meaningful response
    for attempt_num, (variation_type, prompt_to_use) in enumerate(search_variations):
        try:
            logger.info(f"Attempt {attempt_num + 1} ({variation_type}): '{prompt_to_use}'")
            
            response = model.run(
                prompt=prompt_to_use,
                workspace_id=workspace_id,
                user_groups=user_groups,
                images=images,
                documents=documents,
                videos=videos,
                system_prompts=system_prompts,
            )

            logger.debug(f"Response from attempt {attempt_num + 1}: {response}")

            # Extract content and metadata from response
            if isinstance(response, dict):
                content = response.get("content", "")
                metadata = response.get("metadata", {})
            else:
                content = str(response)
                metadata = {}
            
            # Always keep the first response as fallback
            if best_response is None:
                best_response = response
                best_content = content
                best_metadata = metadata
            
            # Check if this is a meaningful response
            if workspace_id:
                # For RAG queries, check if we found information in documents
                if not is_not_found_response(content) and is_meaningful_response(content):
                    logger.info(f"Found meaningful response on attempt {attempt_num + 1}")
                    best_response = response
                    best_content = content
                    best_metadata = metadata
                    break
                else:
                    logger.info(f"Attempt {attempt_num + 1} returned no meaningful results, trying next variation...")
            else:
                # For non-RAG queries, use the first successful response
                if is_meaningful_response(content):
                    logger.info(f"Got meaningful response on attempt {attempt_num + 1}")
                    best_response = response
                    best_content = content
                    best_metadata = metadata
                    break
                
        except Exception as e:
            logger.error(f"Error on attempt {attempt_num + 1}: {str(e)}")
            # Continue to next variation instead of failing
            continue
    
    # Use the best response we found
    content = best_content
    metadata = best_metadata
    
    logger.info(f"Final response selected. Content length: {len(content)}")
    
    # Send final response
    send_to_client(
        {
            "type": "text",
            "action": ChatbotAction.FINAL_RESPONSE.value,
            "timestamp": str(int(round(datetime.now().timestamp()))),
            "userId": user_id,
            "userGroups": user_groups,
            "data": {
                "sessionId": session_id,
                "content": content,
                "metadata": metadata,
                "type": "text"
            },
        }
    )


@tracer.capture_method
def record_handler(record: SQSRecord):
    payload: str = record.body
    message: dict = json.loads(payload)
    detail: dict = json.loads(message["Message"])
    logger.debug(detail)
    logger.info("details", detail=detail)

    if detail["action"] == ChatbotAction.RUN.value:
        handle_run(detail)
    elif detail["action"] == ChatbotAction.HEARTBEAT.value:
        handle_heartbeat(detail)


def handle_failed_records(records):
    for triplet in records:
        status, error, record = triplet
        payload: str = record.body
        message: dict = json.loads(payload)
        detail: dict = json.loads(message["Message"])
        user_id = detail["userId"]
        data = detail.get("data", {})
        session_id = data.get("sessionId", "")

        message_text = "⚠️ *Something went wrong*"
        if (
            "An error occurred (ValidationException)" in error
            and "The provided image must have dimensions in set [1280x720]" in error
        ):
            message_text = "⚠️ *The provided image must have dimensions of 1280x720.*"
        elif (
            "An error occurred (ValidationException)" in error
            and "The width of the provided image must be within range [320, 4096]"
            in error
        ):
            message_text = "⚠️ *The width of the provided image must be within range 320 and 4096 pixels.*"
        elif (
            "An error occurred (AccessDeniedException)" in error
            and "You don't have access to the model with the specified model ID"
            in error
        ):
            message_text = (
                "*This model is not enabled. "
                "Please try again later or contact "
                "an administrator*"
            )
        else:
            logger.error("Unable to process request", error=error)

        send_to_client(
            {
                "type": "text",
                "action": "error",
                "direction": "OUT",
                "userId": user_id,
                "timestamp": str(int(round(datetime.now().timestamp()))),
                "data": {
                    "sessionId": session_id,
                    "content": message_text,
                    "type": "text",
                },
            }
        )


@logger.inject_lambda_context(log_event=False)
@tracer.capture_lambda_handler
def handler(event, context: LambdaContext):
    batch = event["Records"]

    api_keys = parameters.get_secret(API_KEYS_SECRETS_ARN, transform="json")
    for key in api_keys:
        os.environ[key] = api_keys[key]

    try:
        with processor(records=batch, handler=record_handler):
            processed_messages = processor.process()
    except BatchProcessingError as e:
        logger.error(e)

    for message in processed_messages:
        logger.info(
            "Request complete with status " + message[0],
            status=message[0],
            cause=message[1],
        )
    handle_failed_records(
        message for message in processed_messages if message[0] == "fail"
    )

    return processor.response()
