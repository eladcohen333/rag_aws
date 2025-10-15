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


class QuestionProcessor:
    """
    מעבד שאלות - ניקוי, ארגון והעשרה של שאלות בעברית לפני שליחה למודל
    """
    
    def __init__(self):
        # מילות שאלה בעברית בלבד
        self.hebrew_question_words = [
            'מה', 'איך', 'מתי', 'איפה', 'למה', 'מי', 'האם', 'כיצד',
            'מדוע', 'כמה', 'איזה', 'איזו', 'מהו', 'מהי', 'האין'
        ]
        
    def preprocess_question(self, question):
        """
        ניקוי ראשוני של השאלה - הסרת סימני פיסוק מיותרים ורווחים
        """
        import re
        
        if not question or not question.strip():
            return question
            
        # הסרת סימני פיסוק מיותרים אבל שמירה על חיוניים
        question = re.sub(r'[!]{2,}', '!', question)
        question = re.sub(r'[?]{2,}', '?', question)
        question = re.sub(r'[.]{2,}', '.', question)
        
        # הסרת רווחים מיותרים
        question = re.sub(r'\s+', ' ', question).strip()
        
        return question
    
    def generate_search_variations(self, question):
        """
        יצירת 2 וריאציות בלבד של השאלה לחיפוש מקיף
        """
        import re
        
        variations = []
        
        # נסיון 1: השאלה המקורית לאחר ניקוי
        processed = self.preprocess_question(question)
        variations.append(("original", processed))
        
        # נסיון 2: גרסה משופרת - הסרת מילות שאלה והתמקדות במילות מפתח
        enhanced = self._create_enhanced_variation(processed)
        if enhanced and enhanced != processed and len(enhanced.strip()) > 0:
            variations.append(("enhanced", enhanced))
        
        # מוגבל ל-2 נסיונות בלבד
        return variations[:2]
    
    def _create_enhanced_variation(self, question):
        """
        יצירת וריאציה משופרת של השאלה - הסרת מילות שאלה והתמקדות במילות מפתח
        """
        import re
        
        # הסרת מילות שאלה בעברית
        enhanced = question
        for word in self.hebrew_question_words:
            enhanced = re.sub(r'\b' + word + r'\b', '', enhanced, flags=re.IGNORECASE)
        
        # חילוץ מילות מפתח (מילים באורך 2+ תווים)
        words = enhanced.split()
        key_terms = [word for word in words if len(word) >= 2]
        
        if len(key_terms) >= 2:
            # מיון לפי אורך כדי לתת עדיפות למילים חשובות יותר
            sorted_terms = sorted(key_terms, key=len, reverse=True)
            # לקיחת עד 4 מילים חשובות ביותר
            enhanced = ' '.join(sorted_terms[:4])
        else:
            enhanced = ' '.join(key_terms)
        
        # ניקוי רווחים מיותרים
        enhanced = re.sub(r'\s+', ' ', enhanced).strip()
        
        return enhanced if enhanced else question


class ResponseHandler:
    """
    מטפל בתשובות - ניתוח, הערכה וארגון של תשובות המודל
    """
    
    def __init__(self):
        # אינדיקטורים לתשובות "לא נמצא" בעברית בלבד
        self.not_found_indicators = [
            "לא מצאתי במסמך", "לא נמצא במסמך", "אין מידע במסמך",
            "לא קיים במסמך", "המידע לא נמצא", "לא נמצא מידע",
            "אין תוכן רלוונטי", "לא נמצא תוכן", "לא יכול למצוא",
            "אין מידע זמין", "איני יכול לענות על שאלה זו כיוון שהיא אינה מכוסה במסמכים שסופקו",
            "איני יכול לענות על שאלה זו", "השאלה אינה מכוסה במסמכים",
            "אינה מכוסה במסמכים שסופקו", "לא מצאתי מסמכים רלוונטיים"
        ]
        
        # אינדיקטורים להתבססות על מסמכים
        self.document_indicators = [
            "על פי המסמך", "במסמך נכתב", "המסמך מציין",
            "לפי המידע", "בהתאם למסמך", "המידע מציין", "נמצא במסמך",
            "המסמך מתאר", "כפי שמופיע במסמך", "המידע במסמך"
        ]
        
        # תשובות גנריות
        self.generic_responses = [
            "אני לא יודע", "לא יכול לענות", "אין לי מידע",
            "מצטער, אין לי", "לא בטוח"
        ]
        
        # תשובות מעורפלות
        self.vague_responses = [
            "אולי תבדוק", "מומלץ לפנות", "כדאי לברר",
            "יש לוודא", "מומלץ לבדוק"
        ]
    
    def is_not_found_response(self, content):
        """
        בדיקה האם התשובה מציינת שלא נמצא מידע במסמכים
        """
        if not content:
            return False
        
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in self.not_found_indicators)
    
    def has_document_based_content(self, content, metadata):
        """
        בדיקה האם התשובה מבוססת על תוכן מסמכים אמיתי
        """
        if not content or not metadata:
            return False
        
        documents = metadata.get("documents", [])
        if not documents or len(documents) == 0:
            return False
        
        content_lower = content.lower()
        has_document_reference = any(indicator in content_lower for indicator in self.document_indicators)
        
        # בדיקה נוספת - תשובה משמעותית עם מסמכים
        is_substantial = len(content.strip()) > 50
        
        return has_document_reference or (is_substantial and len(documents) > 0)
    
    def is_meaningful_response(self, content):
        """
        בדיקה האם התשובה מכילה מידע משמעותי
        """
        if not content or len(content.strip()) < 10:
            return False
        
        content_lower = content.lower()
        has_generic = any(generic in content_lower for generic in self.generic_responses)
        
        return not has_generic and len(content.strip()) > 20
    
    def is_high_quality_response(self, content, metadata):
        """
        בדיקה האם זו תשובה איכותית שצריכה לעצור את החיפוש
        """
        if not content or not metadata:
            return False
        
        # לא יכולה להיות תשובת "לא נמצא"
        if self.is_not_found_response(content):
            return False
        
        # חייבת להיות משמעותית
        if not self.is_meaningful_response(content):
            return False
        
        # חייבת להיות מבוססת על מסמכים
        if not self.has_document_based_content(content, metadata):
            return False
        
        # בדיקות איכות נוספות
        content_lower = content.lower()
        has_vague_language = any(vague in content_lower for vague in self.vague_responses)
        
        # תשובה איכותית: יש תוכן מסמכים, משמעותית, לא מעורפלת, אורך מספיק
        return not has_vague_language and len(content.strip()) > 80
    
    def generate_no_documents_found_response(self, original_question, attempts_made):
        """
        יצירת תשובה מובנית למקרה של "לא נמצאו מסמכים"
        """
        structured_response = {
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
        return structured_response


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

    # יצירת מעבדי השאלות והתשובות
    question_processor = QuestionProcessor()
    response_handler = ResponseHandler()

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

    # יצירת Pipeline לעיבוד שאלות - מוגבל ל-2 נסיונות בלבד
    best_response = None
    best_content = ""
    best_metadata = {}
    
    # יצירת וריאציות השאלה (מקסימום 2)
    search_variations = question_processor.generate_search_variations(original_prompt)
    
    logger.info(f"Starting streamlined search with {len(search_variations)} attempts (max 2)")
    
    # ביצוע נסיונות חיפוש מוגבלים
    for attempt_num, (variation_type, prompt_to_use) in enumerate(search_variations):
        try:
            logger.info(f"Attempt {attempt_num + 1}/{len(search_variations)} ({variation_type}): '{prompt_to_use}'")
            
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

            # חילוץ תוכן ומטא-דאטה מהתשובה
            if isinstance(response, dict):
                content = response.get("content", "")
                metadata = response.get("metadata", {})
            else:
                content = str(response)
                metadata = {}
            
            # שמירת התשובה הראשונה כגיבוי
            if best_response is None:
                best_response = response
                best_content = content
                best_metadata = metadata
            
            # בדיקת איכות התשובה
            if workspace_id:
                # עבור שאלות RAG - בדיקה קפדנית לתוכן מבוסס מסמכים
                if response_handler.is_high_quality_response(content, metadata):
                    logger.info(f"Found high-quality RAG response on attempt {attempt_num + 1}")
                    best_response = response
                    best_content = content
                    best_metadata = metadata
                    break
                else:
                    # שמירת תשובה טובה יותר אם קיימת
                    if response_handler.is_meaningful_response(content) and (not best_content or len(content) > len(best_content)):
                        logger.info(f"Attempt {attempt_num + 1} - keeping as backup")
                        best_response = response
                        best_content = content
                        best_metadata = metadata
                    else:
                        logger.info(f"Attempt {attempt_num + 1} returned insufficient results")
            else:
                # עבור שאלות רגילות - שימוש בתשובה משמעותית ראשונה
                if response_handler.is_meaningful_response(content):
                    logger.info(f"Got meaningful response on attempt {attempt_num + 1}")
                    best_response = response
                    best_content = content
                    best_metadata = metadata
                    break
                
        except Exception as e:
            logger.error(f"Error on attempt {attempt_num + 1}: {str(e)}")
            continue
    
    # בדיקה אם לא נמצאו מסמכים רלוונטיים ויצירת תשובה מובנית
    if workspace_id and response_handler.is_not_found_response(best_content):
        logger.info("No relevant documents found, generating structured response")
        structured_response = response_handler.generate_no_documents_found_response(
            original_prompt, len(search_variations)
        )
        content = structured_response["content"]
        metadata = structured_response["metadata"]
    else:
        content = best_content
        metadata = best_metadata
    
    logger.info(f"Final response selected. Content length: {len(content)}, Type: {metadata.get('response_type', 'standard')}")
    
    # שליחת התשובה הסופית
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
