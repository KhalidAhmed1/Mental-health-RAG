import os
from dotenv import load_dotenv
from openai import OpenAI
from modules.retriever import retrieve_chunks_hybrid

load_dotenv()

# ── Groq client ─────────────────────────────────────────────────────────────
client_llm = OpenAI(
    api_key  = os.getenv("OPENAI_API_KEY"),
    base_url = os.getenv("OPENAI_BASE_URL"),
)

MAIN_MODEL       = "llama-3.1-8b-instant"
TRANSLATOR_MODEL = "llama-3.1-8b-instant"

# ── Emotion → tone mapping ───────────────────────────────────────────────────
EMOTION_TONE = {
    "sadness" : "Be empathetic and gentle. Acknowledge their pain and show you understand before offering help.",
    "joy"     : "Be warm and encouraging. Celebrate their positive feelings while offering supportive guidance.",
    "fear"    : "Be calm and reassuring. Help ground them and reduce their anxiety with clear, steady guidance.",
    "anger"   : "Be calm and non-confrontational. Validate their frustration without escalating, and guide them gently.",
    "love"    : "Be warm and supportive. Acknowledge their feelings of connection and guide them with care.",
    "surprise": "Be clear and informative. Help them process the unexpected situation with steady, factual support.",
}

# ── Direct response prompts ──────────────────────────────────────────────────
DIRECT_RESPONSE_PROMPTS = {
    "greeting"   : "The user is greeting you. Respond with a warm, friendly greeting and ask how you can help them today.",
    "goodbye"    : "The user is saying goodbye. Respond with a warm farewell and remind them you are here if they need support.",
    "gratitude"  : "The user is thanking you. Respond with a kind acknowledgment and let them know you are always here to help.",
    "out_of_scope": "The user is asking something outside your scope as a mental health assistant. Politely let them know you can only help with mental health related topics and invite them to ask something relevant.",
}


def translate_to_english(text: str, source_language: str) -> str:
    """
    Translates any non-English text to English.
    source_language is a full name e.g. 'Arabic', 'French'.
    """
    response = client_llm.chat.completions.create(
        model    = TRANSLATOR_MODEL,
        messages = [
            {
                "role"   : "system",
                "content": (
                    f"You are a translation engine. "
                    f"Translate the following {source_language} text to English. "
                    f"Return ONLY the translated text — no explanation, "
                    f"no preamble, nothing else."
                ),
            },
            {"role": "user", "content": text},
        ],
        max_tokens  = 512,
        temperature = 0.0,
    )
    return response.choices[0].message.content.strip()


def build_prompt(query: str, chunks: list, emotion: str, language: str):
    """
    Builds the system and user messages for the RAG LLM call.

    Parameters
    ----------
    query    : English version of the user's question
    chunks   : top-k retrieved chunk texts
    emotion  : one of the 6 emotion labels
    language : full language name e.g. 'English', 'Arabic'
    """
    tone = EMOTION_TONE.get(emotion, "Be empathetic and supportive.")

    system_message = f"""You are a compassionate mental health support assistant.
Your role is to provide empathetic, grounded, and helpful responses based ONLY on the context provided.

Tone instruction: {tone}

Rules:
- Answer ONLY from the provided context. Do not invent information.
- If the context does not contain a relevant answer, say so honestly and suggest seeking professional help.
- Keep your response concise, warm, and easy to understand.
- You MUST respond in {language}.
"""

    context_block = "\n\n".join(
        [f"[Context {i+1}]:\n{chunk}" for i, chunk in enumerate(chunks)]
    )

    user_message = f"""Context from knowledge base:
{context_block}

User question: {query}
"""
    return system_message, user_message


def call_llm(system_message: str, user_message: str, history: list = None) -> str:
    """
    Sends the prompt to Groq and returns the response text.
    Injects conversation history between system and user message.
    """
    if history is None:
        history = []

    messages = [{"role": "system", "content": system_message}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = client_llm.chat.completions.create(
        model       = MAIN_MODEL,
        messages    = messages,
        max_tokens  = 1024,
        temperature = 0.7,
    )
    return response.choices[0].message.content.strip()


def get_rag_answer(
    original_query   : str,
    detected_language: str,
    emotion          : str,
    history          : list = None,
    top_k            : int  = 5,
) -> str:
    """
    Full RAG pipeline:
    translate (if needed) → retrieve → build prompt → call LLM
    """
    if history is None:
        history = []

    # Translate to English if input is not English
    if detected_language.lower() != "english":
        english_text = translate_to_english(original_query, detected_language)
    else:
        english_text = original_query

    chunks               = retrieve_chunks_hybrid(english_text, top_k=top_k)
    system_msg, user_msg = build_prompt(english_text, chunks, emotion, detected_language)
    answer               = call_llm(system_msg, user_msg, history)

    return answer


def get_direct_response(intent: str, detected_language: str) -> str:
    """
    Generates a direct response for non-RAG intents
    (greeting, goodbye, gratitude, out_of_scope).
    No history passed — these are simple standalone responses.
    """
    instruction    = DIRECT_RESPONSE_PROMPTS.get(intent, "Respond helpfully and kindly.")
    system_message = f"""You are a compassionate mental health support assistant.
{instruction}
You MUST respond in {detected_language}.
Keep your response short and natural.
"""
    response = client_llm.chat.completions.create(
        model    = MAIN_MODEL,
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": ""},
        ],
        max_tokens  = 256,
        temperature = 0.7,
    )
    return response.choices[0].message.content.strip()
