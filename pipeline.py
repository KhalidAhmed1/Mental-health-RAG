"""
pipeline.py
-----------
Wires all 4 modules into a single pipeline() function.

All models are loaded ONCE at import time (when FastAPI starts),
not on every request. This keeps each request fast.
"""

from modules.language         import predict_language
from modules.llm              import get_rag_answer, get_direct_response, translate_to_english
from modules.intent_classifier import IntentClassifier, get_intent_with_context
from modules.emotion import predict_emotion


# ── Load Module 3 once at startup ────────────────────────────────────────────
# intentExamples.json must be in the project root directory
print("Initializing intent classifier...")
intent_classifier = IntentClassifier()
print("Intent classifier ready.")

# ── Emotion model placeholder ────────────────────────────────────────────────
# Uncomment and update the path once the emotion model pkl is received
# import joblib, os
# _emotion_model = joblib.load(os.path.join("models", "emotion_model.pkl"))
# def predict_emotion(text: str) -> str:
#     return _emotion_model.predict([text])[0]


def pipeline(user_message: str, history: list = None) -> dict:
    """
    Full end-to-end pipeline.

    Parameters
    ----------
    user_message : str  - raw user input (any language)
    history      : list - conversation history as list of
                          {"role": "user"/"assistant", "content": "..."}

    Returns
    -------
    dict with keys: response, intent, emotion, language, history
    """
    if history is None:
        history = []

    # ── Module 1 : Detect language ───────────────────────────────────────────
    detected_language = predict_language(user_message)
    # returns full name: 'English', 'Arabic', 'French' ...

    # ── Translation : non-English → English ─────────────────────────────────
    if detected_language.lower() != "english":
        english_text = translate_to_english(user_message, detected_language)
    else:
        english_text = user_message

    # ── Module 2 : Detect emotion ────────────────────────────────────────────
                             
    emotion = predict_emotion(english_text)

    # ── Module 3 : Detect intent (context-aware) ─────────────────────────────
    intent = get_intent_with_context(english_text, history, intent_classifier)

    # ── Router : RAG or direct response ─────────────────────────────────────
    if intent == "asking_mental_health_question":
        response = get_rag_answer(
            original_query    = user_message,
            detected_language = detected_language,
            emotion           = emotion,
            history           = history,
            top_k             = 5,
        )
    else:
        response = get_direct_response(intent, detected_language)

    # ── Update conversation history ──────────────────────────────────────────
    history.append({"role": "user",      "content": user_message})
    history.append({"role": "assistant", "content": response})

    return {
        "response" : response,
        "intent"   : intent,
        "emotion"  : emotion,
        "language" : detected_language,
        "history"  : history,
    }
