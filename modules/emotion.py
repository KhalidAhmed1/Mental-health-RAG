import os
from transformers import pipeline

_BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR = os.path.join(_BASE_DIR, "models")

# ── Load once at import time ─────────────────────────────────────────────────
print("Loading emotion model...")
_emotion_pipeline = pipeline(
    task      = "text-classification",
    model     = _MODEL_DIR,
    tokenizer = _MODEL_DIR,
    device    = -1,          
    truncation = True,
    max_length = 512,
)
print("Emotion model loaded.")
classes = { 'label_0':'sadness',
    'label_1':'joy',
    'label_2':'love',
    'label_3':'anger',
    'label_4':'fear',
    'label_5':'surprise'}

def predict_emotion(text: str) -> str:
    """
    Returns one of: sadness, joy, fear, anger, love, surprise
    """
    result = _emotion_pipeline(text, truncation=True, max_length=512)
    label  = result[0]["label"].lower()
    return classes[label]