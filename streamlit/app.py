import streamlit as st
import os
import pandas as pd
import numpy as np
import torch
import faiss
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# ========== Constants ==========
ASSETS_BASE_DIR = r"E:\codified\DDReg\LLM_QnA_Chatbot\assets"
CSV_PATH = os.path.join(ASSETS_BASE_DIR, "unique_questions.csv")
GREETING_KEYWORDS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings", "howdy"}
SIMILARITY_THRESHOLD = 0.7

# ========== Model Loading ==========
@st.cache_resource
def load_flan_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

tokenizer, flan_model = load_flan_model()
embed_model = load_embedding_model()

# ========== Utility Functions ==========

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_paths(csv_filepath):
    base_name = os.path.splitext(os.path.basename(csv_filepath))[0]
    folder_path = os.path.join(ASSETS_BASE_DIR, base_name)
    ensure_dir(folder_path)

    return {
        "folder": folder_path,
        "csv": csv_filepath,
        "faiss_index": os.path.join(folder_path, f"{base_name}_faiss.index"),
        "embeddings": os.path.join(folder_path, f"{base_name}_embeddings.npy"),
        "questions": os.path.join(folder_path, f"{base_name}_questions.pkl"),
        "answers": os.path.join(folder_path, f"{base_name}_answers.pkl")
    }

def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def build_index(embeddings, index_path):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index

def load_or_create_files(csv_filepath):
    paths = get_paths(csv_filepath)

    if all(os.path.exists(p) for p in [paths['faiss_index'], paths['embeddings'], paths['questions'], paths['answers']]):
        index = faiss.read_index(paths['faiss_index'])
        embeddings = np.load(paths['embeddings'])
        questions = load_pickle(paths['questions'])
        answers = load_pickle(paths['answers'])
    else:
        df = pd.read_csv(csv_filepath)
        questions = df['question'].tolist()
        answers = df['answer'].tolist()

        embeddings = embed_model.encode(questions)
        save_pickle(questions, paths['questions'])
        save_pickle(answers, paths['answers'])
        np.save(paths['embeddings'], embeddings)
        index = build_index(embeddings, paths['faiss_index'])

    return index, embeddings, questions, answers

def is_greeting(query: str) -> bool:
    return any(greet in query.strip().lower() for greet in GREETING_KEYWORDS)

def flan_llm_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = flan_model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def flan_grammar_correction(text: str) -> str:
    prompt = f"""You are an expert in English grammar and spelling. Correct all spelling mistakes and grammatical errors in the following text and rewrite it as a perfectly phrased English question:\n\n{text}"""
    return flan_llm_response(prompt)

def search_answer(query, index, questions, answers, embeddings, threshold=0.65):
    query_vector = embed_model.encode([query])
    _, indices = index.search(query_vector, k=1)
    matched_index = indices[0][0]
    similarity = cosine_similarity([query_vector[0]], [embeddings[matched_index]])[0][0]

    if similarity >= threshold:
        return questions[matched_index], answers[matched_index], similarity
    return None, None, similarity

def answer_generator(user_input: str):
    if is_greeting(user_input):
        return {"message": " Hello! How can I assist you today?"}

    corrected_query = flan_grammar_correction(user_input)
    index, embeddings, questions, answers = load_or_create_files(CSV_PATH)
    matched_q, matched_a, similarity = search_answer(corrected_query, index, questions, answers, embeddings)

    if matched_q and similarity >= SIMILARITY_THRESHOLD:
        return {
            "corrected_query": corrected_query,
            "matched_question": matched_q,
            "answer": matched_a,
            "similarity": round(float(similarity), 3)
        }
    else:
        return {
            "corrected_query": corrected_query,
            "message": " Sorry, I couldn't find a good answer to your question.",
            "similarity": round(float(similarity), 3)
        }

# ========== Streamlit UI ==========

st.set_page_config(page_title="LLM QnA Chatbot", layout="centered")
st.title("LLM-Powered QnA Chatbot")

query = st.text_input("Enter your question:")
submit = st.button("Submit")

if submit and query:
    with st.spinner("Processing..."):
        response = answer_generator(query)

    st.markdown("### Response:")
    if "corrected_query" in response:
        st.markdown(f"**Corrected Query:** `{response['corrected_query']}`")
    if "matched_question" in response:
        st.markdown(f"**Matched Question:** `{response['matched_question']}`")
        st.markdown(f"**Answer:** {response['answer']}")
        st.markdown(f"**Similarity Score:** `{response['similarity']}`")
    elif "message" in response:
        st.markdown(response["message"])
        if "similarity" in response:
            st.markdown(f"**Similarity Score:** `{response['similarity']}`")
