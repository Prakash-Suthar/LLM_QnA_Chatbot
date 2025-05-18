from app.utils.faissembedding.faissloader import (
    load_embedding_model,
    load_or_create_files,
    search_answer
)   

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load FLAN-T5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
# List of greetings
GREETING_KEYWORDS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings", "howdy"}

def flan_llm_response(prompt: str) -> str:  
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def is_greeting(query: str) -> bool:
    query_lower = query.strip().lower()
    return any(greet in query_lower for greet in GREETING_KEYWORDS)

def flan_grammar_correction(text: str) -> str:
    # Prepare prompt for grammar correction
    prompt = prompt = f"""You are an expert in English grammar and spelling. Correct all spelling mistakes and grammatical errors in the following text and rewrite it as a perfectly phrased English question:

    {text}
    """


    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)

    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def answerGenerator(query: str):
    response = {}

    # Handle greetings first
    if is_greeting(query):
        return {
            "message": "Hello! How can I assist you today?"
        }

    # Load and search in QnA dataset
    csv_path = r"E:\codified\DDReg\LLM_QnA_Chatbot\assets\unique_questions.csv"
    index, embeddings, questions, answers, dataset_name = load_or_create_files(csv_path)
    model_embed = load_embedding_model()

    matched_q, matched_a, score = search_answer(query, model_embed, index, questions, answers, embeddings)

    SIMILARITY_THRESHOLD = 0.7  # adjust as needed

    if matched_q and score >= SIMILARITY_THRESHOLD:
        return {
            "question": matched_q,
            "answer": matched_a,
            "similarity": float(score)
        }
    else:
        # Do not fallback to LLM if no good match
        return {
            "message": "Sorry, I couldn't find a good answer to your question.",
            "similarity": float(score)
        }
