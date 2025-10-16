import os  # Provides access to environment variables like AWS_REGION and secrets
import json  # Used to serialize/deserialize messages between SQS and Lambda
import uuid  # Generates unique session IDs for each chatbot interaction
from datetime import datetime  # Used for timestamps in logging and WebSocket messages
from genai_core.registry import registry  # Handles model adapter registration for Bedrock, OpenAI, etc.
from aws_lambda_powertools import Logger, Tracer  # Provides structured logging and tracing for Lambda observability
from aws_lambda_powertools.utilities import parameters  # Fetches secrets and parameters from AWS Secrets Manager or SSM
from aws_lambda_powertools.utilities.batch import BatchProcessor, EventType  # Simplifies batch processing for SQS-triggered Lambdas
from aws_lambda_powertools.utilities.batch.exceptions import BatchProcessingError  # Exception class for batch processing errors
from aws_lambda_powertools.utilities.data_classes.sqs_event import SQSRecord  # Provides type-safe handling for SQS messages
from aws_lambda_powertools.utilities.typing import LambdaContext  # Typing hint for AWS Lambda context object

import adapters  # noqa: F401 - Required import to register all supported model adapters
from genai_core.utils.websocket import send_to_client  # Helper function to send messages to WebSocket clients in real-time
from genai_core.types import ChatbotAction  # Enum defining possible chatbot actions (RUN, HEARTBEAT, FINAL_RESPONSE, etc.)

# Initialize AWS Lambda Powertools utilities
processor = BatchProcessor(event_type=EventType.SQS)  # Handles SQS batch records efficiently
tracer = Tracer()  # Enables AWS X-Ray distributed tracing
logger = Logger()  # Provides JSON-structured logging with correlation IDs and tracing context

# Retrieve essential environment variables for region and secrets
AWS_REGION = os.environ["AWS_REGION"]  # The AWS region in which the Lambda runs
API_KEYS_SECRETS_ARN = os.environ["API_KEYS_SECRETS_ARN"]  # ARN of the Secrets Manager key storing model API credentials

# Global sequence counter for token streaming order
sequence_number = 0  # Used to preserve token order when streaming partial model outputs


class StreamingAttemptBuffer:
    """Buffers streaming tokens per attempt so the client only sees the chosen response."""

    def __init__(self):
        self._buffers = {}

    def start_attempt(self, attempt_id):
        self._buffers[attempt_id] = []

    def append(self, attempt_id, payload):
        if attempt_id is None:
            return False
        self._buffers.setdefault(attempt_id, []).append(payload)
        return True

    def drain(self, attempt_id):
        return self._buffers.pop(attempt_id, [])

    def discard(self, attempt_id):
        self._buffers.pop(attempt_id, None)

    def clear(self):
        self._buffers.clear()


def on_llm_new_token(user_id, session_id, self, token, run_id, chunk, parent_run_id, *args, **kwargs):
    """
    Handles streaming tokens emitted by the LLM in real-time.
    Sends them back to the connected WebSocket client incrementally.
    """
    if self.disable_streaming:  # If streaming is disabled for this adapter (e.g., some models don't support streaming)
        logger.debug("Streaming is disabled, ignoring token")
        return

    # Handle both LangChain legacy format (string) and new ChatML format (list of dicts)
    if isinstance(token, list):
        text = ""
        for t in token:
            if "text" in t:  # Extract text part from each token dictionary
                text += t.get("text")
    else:
        text = token  # Old-style string token

    if not text:  # Ignore empty tokens
        return

    # Use global sequence number to maintain incremental ordering of tokens
    global sequence_number
    sequence_number += 1

    run_id = str(run_id)  # Ensure run ID is a string (UUID-safe for WebSocket)

    # Build structured WebSocket message with token information
    message_payload = {
        "type": "text",
        "action": ChatbotAction.LLM_NEW_TOKEN.value,  # Action: send incremental token update
        "userId": user_id,
        "timestamp": str(int(round(datetime.now().timestamp()))),  # Current timestamp (epoch seconds)
        "data": {
            "sessionId": session_id,  # The chat session associated with this token stream
            "token": {
                "runId": run_id,  # Unique ID for this inference run
                "sequenceNumber": sequence_number,  # The order index of this token
                "value": text,  # The actual generated token text
            },
        },
    }

    attempt_buffer = getattr(self, "stream_attempt_buffer", None)
    current_attempt = getattr(self, "current_attempt_id", None)
    if attempt_buffer and attempt_buffer.append(current_attempt, message_payload):
        return

    send_to_client(message_payload)


def handle_heartbeat(record):
    """
    Sends periodic heartbeat pings to keep WebSocket connections alive.
    This helps ensure the user session remains active even when no message is being sent.
    """
    user_id = record["userId"]  # Extract user ID from the heartbeat record
    session_id = record["data"]["sessionId"]  # Extract session ID for which heartbeat is triggered

    # Send structured WebSocket heartbeat message
    send_to_client(
        {
            "type": "text",
            "action": ChatbotAction.HEARTBEAT.value,  # Action type defined in ChatbotAction enum
            "timestamp": str(int(round(datetime.now().timestamp()))),
            "userId": user_id,
            "data": {
                "sessionId": session_id,  # Identifies which client connection the ping belongs to
            },
        }
    )


class QuestionProcessor:
    """
    Responsible for preprocessing and enriching Hebrew user questions before sending them to the model.
    Implements a two-step pipeline:
      1. Clean punctuation and whitespace.
      2. Generate a refined version removing generic question words.
    """

    def __init__(self):
        # Common Hebrew question words to remove in the refined version (e.g., "מה", "איך", "למה")
        self.hebrew_question_words = [
            'מה', 'איך', 'מתי', 'איפה', 'למה', 'מי', 'האם', 'כיצד',
            'מדוע', 'כמה', 'איזה', 'איזו', 'מהו', 'מהי', 'האין',
            'תגיד', 'יש', 'האם יש', 'יש איזה'
        ]

    def preprocess_question(self, question: str) -> str:
        """
        Performs light cleaning of the question by removing redundant punctuation and normalizing whitespace.
        Keeps the structure and intent intact.
        """
        import re  # Regular expressions are used for cleanup operations
        if not question or not question.strip():  # Return as-is if question is empty or only whitespace
            return question

        # Replace multiple punctuation marks (e.g., "?!...") with a single space
        question = re.sub(r'[!?.,;:\-]+', ' ', question)

        # Normalize multiple spaces to one, and trim leading/trailing whitespace
        question = re.sub(r'\s+', ' ', question).strip()

        return question  # Return the cleaned version of the question

    def _create_enhanced_variation(self, question: str) -> str:
        """
        Produces a refined version of the question by removing generic Hebrew question words.
        This helps focus retrieval on meaningful content (keywords) rather than phrasing.
        Example:
            Input:  "תגיד, יש איזה שינוי מיוחד בטופס 106 השנה לגבי השכר במגזר הציבורי?"
            Output: "יש שינוי מיוחד בטופס 106 השנה לגבי השכר במגזר הציבורי"
        """
        import re
        enhanced = question  # Start with the cleaned question as base

        # Remove common question words if they appear as standalone tokens
        for word in self.hebrew_question_words:
            pattern = rf"\b{word}\b"  # Word-boundary ensures partial matches (e.g., "מהות") aren’t removed
            enhanced = re.sub(pattern, '', enhanced, flags=re.IGNORECASE)

        # Clean up redundant spaces after removal
        enhanced = re.sub(r'\s+', ' ', enhanced).strip()

        # If the result becomes too short (less than 5 words), revert to the original question
        if len(enhanced.split()) < 5:
            return question

        return enhanced

    def generate_search_variations(self, question: str):
        """
        Generates up to two query variations for the search pipeline:
          1. cleaned – punctuation/spacing normalized.
          2. refined – generic Hebrew question words removed for semantic precision.
        Returns:
            List of tuples: [(“cleaned”, str), (“refined”, str)]
        """
        variations = []  # Holds the tuple list (variation_type, text)

        # Step 1: Generate cleaned version
        cleaned = self.preprocess_question(question)
        variations.append(("cleaned", cleaned))

        # Step 2: Generate refined version (if different)
        enhanced = self._create_enhanced_variation(cleaned)
        if enhanced != cleaned:
            variations.append(("refined", enhanced))

        # Limit to a maximum of two attempts
        return variations[:2]
class ResponseHandler:
    """
    Handles model responses:
    - Determines if a response is document-based, meaningful, or generic.
    - Evaluates quality to decide whether to retry or finalize.
    - Generates fallback responses when no documents are found.
    """

    def __init__(self):
        # Common Hebrew phrases that indicate the model didn’t find relevant data in documents
        self.not_found_indicators = [
            "לא מצאתי במסמך", "לא נמצא במסמך", "אין מידע במסמך",
            "לא קיים במסמך", "המידע לא נמצא", "לא נמצא מידע",
            "אין תוכן רלוונטי", "לא נמצא תוכן", "לא יכול למצוא",
            "אין מידע זמין", "איני יכול לענות על שאלה זו כיוון שהיא אינה מכוסה במסמכים שסופקו",
            "איני יכול לענות על שאלה זו", "השאלה אינה מכוסה במסמכים",
            "אינה מכוסה במסמכים שסופקו", "לא מצאתי מסמכים רלוונטיים"
        ]

        # Indicators that a response explicitly references a document
        self.document_indicators = [
            "על פי המסמך", "במסמך נכתב", "המסמך מציין",
            "לפי המידע", "בהתאם למסמך", "המידע מציין", "נמצא במסמך",
            "המסמך מתאר", "כפי שמופיע במסמך", "המידע במסמך"
        ]

        # Generic fallback phrases meaning “I don’t know”
        self.generic_responses = [
            "אני לא יודע", "לא יכול לענות", "אין לי מידע",
            "מצטער, אין לי", "לא בטוח"
        ]

        # Vague or non-committal responses
        self.vague_responses = [
            "אולי תבדוק", "מומלץ לפנות", "כדאי לברר",
            "יש לוודא", "מומלץ לבדוק"
        ]

    def is_not_found_response(self, content):
        """Checks if the response contains any known “no data found” indicators."""
        if not content:
            return False
        content_lower = content.lower()
        return any(ind in content_lower for ind in self.not_found_indicators)

    def has_document_based_content(self, content, metadata):
        """
        Determines whether the response actually references source documents.
        This ensures the model grounded its answer in retrieved evidence.
        """
        if not content or not metadata:
            return False
        documents = metadata.get("documents", [])
        if not documents:
            return False
        content_lower = content.lower()
        has_reference = any(ind in content_lower for ind in self.document_indicators)
        # Consider also responses with sufficient length and attached documents as valid
        return has_reference or (len(content.strip()) > 50 and len(documents) > 0)

    def is_meaningful_response(self, content):
        """Checks if the model’s response is long enough and not purely generic."""
        if not content or len(content.strip()) < 10:
            return False
        content_lower = content.lower()
        has_generic = any(g in content_lower for g in self.generic_responses)
        return not has_generic and len(content.strip()) > 20

    def is_high_quality_response(self, content, metadata):
        """
        Combines multiple checks to determine if the response should end the pipeline:
        - Must not be a “not found” message
        - Must be meaningful
        - Must be document-grounded
        - Must not be vague
        """
        if not content or not metadata:
            return False
        if self.is_not_found_response(content):
            return False
        if not self.is_meaningful_response(content):
            return False
        if not self.has_document_based_content(content, metadata):
            return False
        content_lower = content.lower()
        has_vague = any(v in content_lower for v in self.vague_responses)
        return not has_vague and len(content.strip()) > 80

    def get_failure_reason(self, content, metadata):
        """
        Returns a short reason explaining why a response failed the quality check.
        Used for debug logs to improve traceability.
        """
        if not content:
            return "empty"
        if self.is_not_found_response(content):
            return "not_found"
        if not self.is_meaningful_response(content):
            return "unmeaningful"
        if not self.has_document_based_content(content, metadata):
            return "no_documents"
        content_lower = content.lower()
        if any(v in content_lower for v in self.vague_responses):
            return "vague"
        return "unknown"

    def generate_no_documents_found_response(self, original_question, attempts_made):
        """
        Builds a structured fallback response for cases where no relevant
        documents were found across all search attempts.
        """
        return {
            "content": f"לא מצאתי מסמכים רלוונטיים לשאלה '{original_question}' במאגר הנתונים הנוכחי.",
            "metadata": {
                "response_type": "no_documents_found",
                "original_question": original_question,
                "search_attempts": attempts_made,
                "suggestions": [
                    "נסה לנסח את השאלה בצורה שונה",
                    "השתמש במילות מפתח אחרות",
                    "בדוק אם המסמכים הרלוונטיים הועלו למערכת"
                ],
                "documents": [],
                "timestamp": str(int(round(datetime.now().timestamp())))
            }
        }


def handle_run(record):
    """
    Main orchestration function that handles incoming chat requests.
    Executes the full query lifecycle:
    1. Preprocess question → generate variations.
    2. Run model inference per variation.
    3. Evaluate response quality.
    4. Send best result to WebSocket.
    """
    # Extract basic request context
    user_id = record["userId"]
    user_groups = record["userGroups"]
    data = record["data"]

    provider = data["provider"]  # Model provider (bedrock, openai, etc.)
    model_id = data["modelName"]  # Specific model identifier
    mode = data["mode"]  # Generation mode (e.g., chat, completion)
    original_prompt = data["text"]  # Original user question
    workspace_id = data.get("workspaceId", None)  # Used for RAG mode
    session_id = data.get("sessionId") or str(uuid.uuid4())  # Ensure session always has a UUID
    images = data.get("images", [])  # Optional image inputs
    documents = data.get("documents", [])  # Optional retrieved docs
    videos = data.get("videos", [])  # Optional video inputs
    system_prompts = record.get("systemPrompts", {})  # System-level context (instruction prompts)

    # Initialize question and response handlers
    question_processor = QuestionProcessor()
    response_handler = ResponseHandler()

    # Retrieve appropriate adapter for the model provider (e.g., BedrockAdapter)
    adapter = registry.get_adapter(f"{provider}.{model_id}")

    # Connect the adapter’s streaming token callback to WebSocket sender
    adapter.on_llm_new_token = lambda *args, **kwargs: on_llm_new_token(user_id, session_id, *args, **kwargs)

    # Initialize the model instance with current context
    model = adapter(
        model_id=model_id,
        mode=mode,
        session_id=session_id,
        user_id=user_id,
        model_kwargs=data.get("modelKwargs", {}),
    )

    model.stream_attempt_buffer = StreamingAttemptBuffer()
    model.current_attempt_id = None

    # Variables for best response tracking
    best_response = None
    best_content = ""
    best_metadata = {}
    best_attempt_index = None

    # Generate cleaned/refined question variations (max 2)
    search_variations = question_processor.generate_search_variations(original_prompt)
    logger.info(f"Starting streamlined search with {len(search_variations)} attempts (max 2)")

    # Iterate through variations sequentially
    for attempt_num, (variation_type, prompt_to_use) in enumerate(search_variations):
        model.current_attempt_id = attempt_num
        model.stream_attempt_buffer.start_attempt(attempt_num)

        try:
            logger.info(f"Attempt {attempt_num + 1}/{len(search_variations)} ({variation_type}): '{prompt_to_use}'")

            # Execute the model run for this variation
            response = model.run(
                prompt=prompt_to_use,
                workspace_id=workspace_id,
                user_groups=user_groups,
                images=images,
                documents=documents,
                videos=videos,
                system_prompts=system_prompts,
            )

            # Normalize model output to unified dict format
            if isinstance(response, dict):
                content = response.get("content", "")
                metadata = response.get("metadata", {})
            else:
                content = str(response)
                metadata = {}

            # Store the first response as a fallback baseline
            if best_response is None:
                best_response, best_content, best_metadata = response, content, metadata
                best_attempt_index = attempt_num

            # Evaluate response quality depending on RAG mode
            if workspace_id:
                # For RAG-based queries → must be document-grounded and meaningful
                if response_handler.is_high_quality_response(content, metadata):
                    logger.info(f"Found high-quality RAG response on attempt {attempt_num + 1}")
                    best_response, best_content, best_metadata = response, content, metadata
                    best_attempt_index = attempt_num
                    break  # Stop pipeline early (good answer)
                else:
                    reason = response_handler.get_failure_reason(content, metadata)
                    logger.info(f"Attempt {attempt_num + 1} failed reason: {reason}")
            else:
                # For normal chat mode → only need meaningful text
                if response_handler.is_meaningful_response(content):
                    logger.info(f"Got meaningful non-RAG response on attempt {attempt_num + 1}")
                    best_response, best_content, best_metadata = response, content, metadata
                    best_attempt_index = attempt_num
                    break
                else:
                    reason = response_handler.get_failure_reason(content, metadata)
                    logger.info(f"Attempt {attempt_num + 1} failed reason: {reason}")

        except Exception as e:
            # Log any runtime error per attempt without halting the pipeline
            logger.error(f"Error on attempt {attempt_num + 1}: {str(e)}")
            model.stream_attempt_buffer.discard(attempt_num)
            continue

    model.current_attempt_id = None

    # Handle final fallback: if all attempts failed or no docs found
    use_structured_response = False
    if workspace_id and response_handler.is_not_found_response(best_content):
        logger.info("No relevant documents found, generating structured response")
        structured = response_handler.generate_no_documents_found_response(original_prompt, len(search_variations))
        content, metadata = structured["content"], structured["metadata"]
        use_structured_response = True
    else:
        content, metadata = best_content, best_metadata

    if not use_structured_response and best_attempt_index is not None:
        for token_message in model.stream_attempt_buffer.drain(best_attempt_index):
            send_to_client(token_message)
        model.stream_attempt_buffer.clear()
    else:
        model.stream_attempt_buffer.clear()

    logger.info(f"Final response selected. Content length: {len(content)}, Type: {metadata.get('response_type', 'standard')}")

    # Send the final message to the user via WebSocket
    send_to_client({
        "type": "text",
        "action": ChatbotAction.FINAL_RESPONSE.value,
        "timestamp": str(int(round(datetime.now().timestamp()))),
        "userId": user_id,
        "userGroups": user_groups,
        "data": {
            "sessionId": session_id,
            "content": content,
            "metadata": metadata,
            "type": "text",
        },
    })


@tracer.capture_method
def record_handler(record: SQSRecord):
    """
    Processes a single SQS record in the Lambda batch.
    Determines the chatbot action and dispatches to the correct handler.
    """
    payload = record.body  # Raw SQS message body
    message = json.loads(payload)  # Deserialize outer SNS wrapper
    detail = json.loads(message["Message"])  # Deserialize inner event payload

    logger.info("details", detail=detail)  # Log contextual info for debugging

    # Dispatch action type
    if detail["action"] == ChatbotAction.RUN.value:
        handle_run(detail)
    elif detail["action"] == ChatbotAction.HEARTBEAT.value:
        handle_heartbeat(detail)


def handle_failed_records(records):
    """
    Sends error notifications via WebSocket for any failed SQS messages.
    Provides user-friendly feedback based on common error patterns.
    """
    for status, error, record in records:
        payload = record.body
        message = json.loads(payload)
        detail = json.loads(message["Message"])
        user_id = detail["userId"]
        data = detail.get("data", {})
        session_id = data.get("sessionId", "")

        # Default generic error message
        message_text = "⚠️ *Something went wrong*"

        # Match specific validation errors for more precise user feedback
        if "ValidationException" in error and "dimensions in set [1280x720]" in error:
            message_text = "⚠️ *The provided image must have dimensions of 1280x720.*"
        elif "ValidationException" in error and "width of the provided image" in error:
            message_text = "⚠️ *The width of the provided image must be between 320 and 4096 pixels.*"
        elif "AccessDeniedException" in error and "model with the specified model ID" in error:
            message_text = "*This model is not enabled. Please try again later or contact an administrator*"
        else:
            logger.error("Unable to process request", error=error)

        # Send structured error response to the user
        send_to_client({
            "type": "text",
            "action": "error",
            "direction": "OUT",
            "userId": user_id,
            "timestamp": str(int(round(datetime.now().timestamp()))),
            "data": {"sessionId": session_id, "content": message_text, "type": "text"},
        })


@logger.inject_lambda_context(log_event=False)
@tracer.capture_lambda_handler
def handler(event, context: LambdaContext):
    """
    Main AWS Lambda entrypoint for SQS-triggered RAG chatbot requests.
    Handles batching, secrets retrieval, and error reporting.
    """
    batch = event["Records"]  # List of SQS messages in the batch

    # Load API keys or model credentials from Secrets Manager
    api_keys = parameters.get_secret(API_KEYS_SECRETS_ARN, transform="json")
    for key, value in api_keys.items():
        os.environ[key] = value  # Inject secrets into environment for adapters

    try:
        # Process batch using the configured SQS processor
        with processor(records=batch, handler=record_handler):
            processed_messages = processor.process()
    except BatchProcessingError as e:
        logger.error(e)

    # Log result summary for all processed messages
    for message in processed_messages:
        logger.info("Request complete with status " + message[0], status=message[0], cause=message[1])

    # Notify users about failed message deliveries
    handle_failed_records(msg for msg in processed_messages if msg[0] == "fail")

    return processor.response()  # Return batch result summary to AWS Lambda
