# import os
# import joblib

# _BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# _emotion_model = joblib.load(os.path.join(_BASE_DIR, "models", "emotion_model.pkl"))

# def predict_emotion(text: str) -> str:
#     """
#     Returns one of: sadness, joy, fear, anger, love, surprise
#     """
#     return _emotion_model.predict([text])[0]