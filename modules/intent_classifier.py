import os
import json
import logging
import sys
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.prompts import (
    PromptTemplate,
    FewShotPromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_groq import ChatGroq

load_dotenv()

# ── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt  = "%H:%M:%S",
    handlers = [logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an advanced triage and routing assistant for a specialized mental health support system. 
Your sole responsibility is to analyze the user's input and categorize it into exactly ONE of the allowed intent classes.

CRITICAL INSTRUCTIONS:
1. Base your classification entirely on the core intent of the user's statement.
2. Rely heavily on the few-shot examples provided below to understand the classification boundaries.
3. Choose exactly one of the allowed categories. Do not invent new categories.

ALLOWED CATEGORIES AND DEFINITIONS:

- greeting: 
  The user is starting a conversation, saying hello, or checking if someone is online.
  (e.g., "Hi", "Hello", "Is anyone there?", "Good morning")

- goodbye: 
  The user is ending the conversation, signing off, or indicating they are leaving.
  (e.g., "Bye", "See you later", "Talk to you tomorrow", "I'm heading out")

- gratitude: 
  The user is thanking the system, expressing appreciation, or confirming that their issue was resolved.
  (e.g., "Thank you so much", "Thanks for the help", "That makes sense, thank you", "Appreciate it")

- asking_mental_health_question: 
  The user is actively seeking help, coping mechanisms, definitions, or advice regarding mental health conditions, emotions, symptoms, or psychological well-being. This is a critical class that triggers our clinical knowledge retrieval pipeline.
  (e.g., "How do I deal with panic attacks?", "I'm feeling incredibly anxious right now", "What are the signs of burnout?", "Can you give me tips for depression?")

- out_of_scope: 
  The user is asking about general knowledge, coding, math, recipes, casual chit-chat, or anything completely unrelated to mental health support.
  (e.g., "What is the capital of France?", "Write a python script", "Tell me a joke", "How's the weather?")

Analyze the context carefully. If a user says "Hello, I am having a panic attack", the primary intent is 'asking_mental_health_question', not 'greeting'. Prioritize clinical inquiries over conversational fluff.
"""

# ── Pydantic output schema ───────────────────────────────────────────────────
class IntentResponse(BaseModel):
    intent: Literal[
        "greeting",
        "goodbye",
        "gratitude",
        "asking_mental_health_question",
        "out_of_scope",
    ] = Field(description="The classified intent of the user's message.")


# ── Classifier class ─────────────────────────────────────────────────────────
class IntentClassifier:

    def __init__(
        self,
        model_name  : str   = "llama-3.3-70b-versatile",
        temperature : float = 0,
    ):
        logger.info(f"Initializing IntentClassifier with model='{model_name}', temp={temperature}")
        self.model_name  = model_name
        self.temperature = temperature
        self.examples    = self._load_examples()
        self.chain       = self._build_chain()
        logger.info("IntentClassifier initialization complete.")

    def _load_examples(self):
        # intentExamples.json is in the project root directory
        _base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        examples_path = os.path.join(_base_dir, "intentExamples.json")

        logger.info(f"Loading few-shot examples from '{examples_path}'...")
        try:
            with open(examples_path, "r", encoding="utf-8") as f:
                examples = json.load(f)
            logger.info(f"Successfully loaded {len(examples)} few-shot examples.")
            return examples
        except Exception as e:
            logger.error(f"Failed to load examples file: {str(e)}")
            raise

    def _build_chain(self):
        logger.info("Assembling LangChain LCEL pipeline components...")

        example_prompt = PromptTemplate(
            input_variables = ["query", "intent"],
            template        = "User: {query}\nIntent: {intent}",
        )

        few_shot_prompt = FewShotPromptTemplate(
            examples        = self.examples,
            example_prompt  = example_prompt,
            prefix          = SYSTEM_PROMPT,
            suffix          = "User: {input}\nIntent:",
            input_variables = ["input"],
        )

        system_message_prompt = SystemMessagePromptTemplate(prompt=few_shot_prompt)
        human_message_prompt  = HumanMessagePromptTemplate.from_template("{input}")
        chat_prompt           = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        logger.info("Prompt templates structured successfully.")

        logger.info(f"Connecting to Groq API for model '{self.model_name}'...")
        llm = ChatGroq(
            model       = self.model_name,
            temperature = self.temperature,
            api_key     = os.getenv("OPENAI_API_KEY"),
        )

        logger.info("Binding Pydantic output schema (IntentResponse) to LLM...")
        structured_llm = llm.with_structured_output(IntentResponse)

        chain = chat_prompt | structured_llm
        logger.info("LCEL routing chain compiled successfully.")
        return chain

    def predict(self, text: str) -> str:
        logger.info(f"Received user input for prediction: '{text}'")
        logger.info("Invoking LLM chain...")

        try:
            prediction: IntentResponse = self.chain.invoke({"input": text})
            logger.info(f"Extracted intent: '{prediction.intent}'")
            return prediction.intent
        except Exception as e:
            logger.error(f"Error during chain execution: {str(e)}")
            raise


# ── Context-aware intent detection ───────────────────────────────────────────
def get_intent_with_context(
    english_text      : str,
    history           : list,
    intent_classifier : IntentClassifier,
) -> str:
    """
    Passes the previous user message as context when classifying follow-up
    messages like 'What can you do about it?' so they are not classified
    as out_of_scope.
    """
    if history and len(history) >= 2:
        last_user_msg = history[-2]["content"]
        context_input = (
            f"Previous message: {last_user_msg}\n"
            f"Current message: {english_text}"
        )
    else:
        context_input = english_text

    return intent_classifier.predict(context_input)