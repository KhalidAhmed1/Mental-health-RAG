import os
import joblib

# ── Language code → full name mapping ──────────────────────────────────────
# Matches exactly what your language SVC model outputs (2-letter ISO codes)
LANGUAGE_NAMES = {
    "pt": "Portuguese",
    "bg": "Bulgarian",
    "zh": "Chinese",
    "th": "Thai",
    "ru": "Russian",
    "pl": "Polish",
    "ur": "Urdu",
    "sw": "Swahili",
    "tr": "Turkish",
    "es": "Spanish",
    "ar": "Arabic",
    "it": "Italian",
    "hi": "Hindi",
    "de": "German",
    "el": "Greek",
    "nl": "Dutch",
    "fr": "French",
    "vi": "Vietnamese",
    "en": "English",
    "ja": "Japanese",
}

# ── Load language detection models once at import time ─────────────────────
_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_svc_model  = joblib.load(os.path.join(_BASE_DIR, "models", "language_svc_model.pkl"))
_vectorizer = joblib.load(os.path.join(_BASE_DIR, "models", "tfidf_vectorizer.pkl"))


def predict_language(text: str) -> str:
    """
    Detects the language of the input text.
    Returns a full language name e.g. 'English', 'Arabic', 'French'.
    """
    text_vec   = _vectorizer.transform([text])
    prediction = _svc_model.predict(text_vec)
    return LANGUAGE_NAMES.get(prediction[0], prediction[0])