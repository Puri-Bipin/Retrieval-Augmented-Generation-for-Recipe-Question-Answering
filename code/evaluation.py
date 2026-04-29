#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for RAG outputs.
Computes faithfulness, answer relevance, retrieval precision, and recall.
Requires the three CSV files (phi3, mistral, llama3) and the original recipe texts.
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# Paths (adjust as needed)
# ============================================================
RECIPE_FOLDER = "data/recipes/"
RESULTS_FOLDER = "data/results/"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 6

# Ground truth source recipe for each question (must match the order in QUESTIONS)
CORRECT_SOURCE_RECIPE = [
    "Chicken Caesar Wrap",
    "chocolate_chip_scones",
    "Crispy Breaded Cauliflower Nuggets",
    "3-Ingredient Cajun Alfredo Pasta",
    "Easy Roast Chicken",
    "Divorce Carrot Cake",
    "Easy Lasagna Primavera",
    "Easy Roast Chicken",
    "How To Make a Burger on the Stove",
    "How To Make Donuts",
    "Malted Chocolate Brownies",
    "One-Bowl Lemon Snack Cake",
    "Quick Blueberry Muffins"
]

# ============================================================
# Helpers to rebuild index (same as in pipeline)
# ============================================================
def build_index(recipe_folder):
    recipe_docs = []
    recipe_names = []
    for fname in os.listdir(recipe_folder):
        if fname.endswith(".txt"):
            with open(os.path.join(recipe_folder, fname), 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:
                    recipe_docs.append(text)
                    recipe_names.append(fname.replace(".txt", ""))
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_chunks = []
    metadata = []
    for idx, doc in enumerate(recipe_docs):
        chunks = splitter.split_text(doc)
        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append({"recipe_id": idx, "recipe_name": recipe_names[idx]})
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    chunk_embs = embed_model.encode(all_chunks, show_progress_bar=True)
    dim = chunk_embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(chunk_embs).astype('float32'))
    return all_chunks, metadata, embed_model, index

def retrieve_relevant_chunks(query, all_chunks, metadata, embed_model, index, top_k=TOP_K_RETRIEVAL):
    q_emb = embed_model.encode([query])
    distances, indices = index.search(np.array(q_emb).astype('float32'), top_k)
    return [{"text": all_chunks[idx], "recipe": metadata[idx]["recipe_name"]} for idx in indices[0]]

# ============================================================
# Metric functions
# ============================================================
def faithfulness_score(answer, retrieved_chunks):
    """Proportion of content words in answer that appear in retrieved chunks."""
    if not answer or answer.startswith("ERROR"):
        return 0.0
    words = set(answer.lower().split())
    stopwords = {"the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "by", "with", "without", "is", "are", "was", "were", "be", "been", "being"}
    content_words = words - stopwords
    if not content_words:
        return 0.0
    all_chunk_text = " ".join([chunk["text"].lower() for chunk in retrieved_chunks])
    matched = sum(1 for w in content_words if w in all_chunk_text)
    return matched / len(content_words)

def answer_relevance(answer, question, embed_model):
    """Cosine similarity between question and answer embeddings."""
    if not answer or answer.startswith("ERROR"):
        return 0.0
    emb_q = embed_model.encode([question])
    emb_a = embed_model.encode([answer])
    return float(cosine_similarity(emb_q, emb_a)[0][0])

def retrieval_precision(sources_str, correct_recipe):
    """Among retrieved recipes, fraction that are the correct one."""
    retrieved_list = [r.strip() for r in sources_str.split(",")]
    if not retrieved_list:
        return 0.0
    correct_count = sum(1 for r in retrieved_list if r == correct_recipe)
    return correct_count / len(retrieved_list)

def retrieval_recall(sources_str, correct_recipe):
    """Whether the correct recipe was retrieved at least once."""
    retrieved_list = [r.strip() for r in sources_str.split(",")]
    return 1.0 if correct_recipe in retrieved_list else 0.0

# ============================================================
# Evaluate one model's CSV
# ============================================================
def evaluate_model(df, model_name, all_chunks, metadata, embed_model, index):
    results = []
    for idx, row in df.iterrows():
        q = row["question"]
        ans = row["answer"]
        sources_str = row["sources"]
        correct = CORRECT_SOURCE_RECIPE[idx]

        retrieved_chunks = retrieve_relevant_chunks(q, all_chunks, metadata, embed_model, index)
        faith = faithfulness_score(ans, retrieved_chunks)
        rel = answer_relevance(ans, q, embed_model)
        prec = retrieval_precision(sources_str, correct)
        rec = retrieval_recall(sources_str, correct)

        results.append({
            "model": model_name,
            "question": q,
            "answer": ans,
            "faithfulness": faith,
            "answer_relevance": rel,
            "retrieval_precision": prec,
            "retrieval_recall": rec,
            "correct_source": correct,
            "retrieved_sources": sources_str
        })
    return pd.DataFrame(results)

# ============================================================
# Main evaluation
# ============================================================
def evaluate_all_models():
    print("Building FAISS index from recipes...")
    all_chunks, metadata, embed_model, index = build_index(RECIPE_FOLDER)

    # Load CSVs (use latin-1 encoding to avoid degree symbol issues)
    df_phi = pd.read_csv(os.path.join(RESULTS_FOLDER, "rag_result_phi3.csv"), encoding='latin-1')
    df_mistral = pd.read_csv(os.path.join(RESULTS_FOLDER, "rag_result_mistral.csv"), encoding='latin-1')
    df_llama = pd.read_csv(os.path.join(RESULTS_FOLDER, "rag_result_llama3.csv"), encoding='latin-1')

    print("Evaluating Phi-3...")
    df_phi_eval = evaluate_model(df_phi, "phi3", all_chunks, metadata, embed_model, index)
    print("Evaluating Mistral...")
    df_mistral_eval = evaluate_model(df_mistral, "mistral", all_chunks, metadata, embed_model, index)
    print("Evaluating Llama-3...")
    df_llama_eval = evaluate_model(df_llama, "llama3", all_chunks, metadata, embed_model, index)

    all_eval = pd.concat([df_phi_eval, df_mistral_eval, df_llama_eval], ignore_index=True)

    # Summary averages
    summary = all_eval.groupby("model").agg({
        "faithfulness": "mean",
        "answer_relevance": "mean",
        "retrieval_precision": "mean",
        "retrieval_recall": "mean"
    }).round(4)
    print("\n=== Average Metrics per Model ===")
    print(summary)

    # Save results
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    all_eval.to_csv(os.path.join(RESULTS_FOLDER, "detailed_metrics_all_models.csv"), index=False)
    summary.to_csv(os.path.join(RESULTS_FOLDER, "summary_metrics.csv"))
    print("\nResults saved to", RESULTS_FOLDER)

if __name__ == "__main__":
    evaluate_all_models()