import os
import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ASSETS_BASE_DIR = r"E:\codified\DDReg\LLM_QnA_Chatbot\assets"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_paths(csv_filepath):
    base_name = os.path.splitext(os.path.basename(csv_filepath))[0]
    folder_path = os.path.join(ASSETS_BASE_DIR, base_name)
    ensure_dir(folder_path)

    paths = {
        "folder": folder_path,
        "csv": csv_filepath,
        "faiss_index": os.path.join(folder_path, f"{base_name}_faiss.index"),
        "embeddings": os.path.join(folder_path, f"{base_name}_embeddings.npy"),
        "questions": os.path.join(folder_path, f"{base_name}_questions.pkl"),
        "answers": os.path.join(folder_path, f"{base_name}_answers.pkl")
    }
    return paths, base_name

def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def build_index(embeddings, index_path):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index

def load_or_create_files(csv_filepath):
    paths, name = get_paths(csv_filepath)

    # Check if all necessary files exist
    if all(os.path.exists(p) for p in [paths['faiss_index'], paths['embeddings'], paths['questions'], paths['answers']]):
        print(f"[INFO] Reusing existing assets in: {paths['folder']}")
        index = faiss.read_index(paths['faiss_index'])
        embeddings = np.load(paths['embeddings'])
        questions = load_pickle(paths['questions'])
        answers = load_pickle(paths['answers'])
    else:
        print(f"[INFO] Generating assets for: {csv_filepath}")
        df = pd.read_csv(csv_filepath)
        questions = df['question'].tolist()
        answers = df['answer'].tolist()

        model = load_embedding_model()
        embeddings = model.encode(questions)

        # Save everything
        save_pickle(questions, paths['questions'])
        save_pickle(answers, paths['answers'])
        np.save(paths['embeddings'], embeddings)
        index = build_index(embeddings, paths['faiss_index'])

    return index, embeddings, questions, answers, name

def search_answer(query, model, index, questions, answers, embeddings, threshold=0.65):
    query_vector = model.encode([query])
    _, indices = index.search(query_vector, k=1)
    matched_index = indices[0][0]

    similarity = cosine_similarity([query_vector[0]], [embeddings[matched_index]])[0][0]

    if similarity >= threshold:
        return questions[matched_index], answers[matched_index], similarity
    else:
        return None, None, similarity

