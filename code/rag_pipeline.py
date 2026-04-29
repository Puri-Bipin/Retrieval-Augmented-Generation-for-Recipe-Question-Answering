#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Pipeline for Recipe Question Answering
Loads a model (Phi-3, Mistral, or Llama-3), builds a FAISS index from recipe texts,
and answers a set of predefined questions. Saves results to CSV.
"""

import os
import torch
import pandas as pd
import numpy as np
from unsloth import FastLanguageModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss

# ============================================================
# Configuration (modify as needed)
# ============================================================
RECIPE_FOLDER = "data/recipes/"          # Folder with .txt recipe files
RESULTS_FOLDER = "data/results/"         # Where to save CSV outputs
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 6
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.1

# List of 13 domain-specific questions
QUESTIONS = [
    "How many days can the chicken Caesar wraps be refrigerated?",
    "What is the purpose of chilling the scones before baking?",
    "What ingredient is used to make the breading mixture 'cheesy' without cheese?",
    "How many calories are in one serving of the 3-Ingredient Cajun Alfredo Pasta?",
    "What is the protein content per serving in the Easy Roast Chicken?",
    "How long should the cake cool before frosting?",
    "What is the purpose of spraying aluminum foil before covering the lasagna?",
    "At what internal temperature does the USDA recommend the chicken breast to be fully cooked?",
    "Why do you press down on each patty with a spatula for 10 seconds after placing it in the skillet?",
    "What is the recommended oil temperature for frying the donuts?",
    "How can you prolong the moistness of the brownies after cooling?",
    "On which day does the cake have the best flavor according to the recipe?",
    "What two oven temperatures are used to bake the muffins, and for how long at each?"
]

def build_index(recipe_folder):
    """Load recipe .txt files, chunk them, and build FAISS index."""
    recipe_docs = []
    recipe_names = []
    for fname in os.listdir(recipe_folder):
        if fname.endswith(".txt"):
            with open(os.path.join(recipe_folder, fname), 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:
                    recipe_docs.append(text)
                    recipe_names.append(fname.replace(".txt", ""))
    print(f"Loaded {len(recipe_docs)} recipes")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    all_chunks = []
    metadata = []
    for idx, doc in enumerate(recipe_docs):
        chunks = splitter.split_text(doc)
        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append({"recipe_id": idx, "recipe_name": recipe_names[idx]})
    print(f"Created {len(all_chunks)} chunks")

    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    chunk_embs = embed_model.encode(all_chunks, show_progress_bar=True)
    dim = chunk_embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(chunk_embs).astype('float32'))
    return all_chunks, metadata, embed_model, index

def retrieve_relevant_chunks(query, all_chunks, metadata, embed_model, index, top_k=TOP_K_RETRIEVAL):
    """Retrieve top_k text chunks for a query."""
    q_emb = embed_model.encode([query])
    distances, indices = index.search(np.array(q_emb).astype('float32'), top_k)
    return [{"text": all_chunks[idx], "recipe": metadata[idx]["recipe_name"]} for idx in indices[0]]

def load_model(model_name):
    """Load a model by name: 'phi3', 'mistral', or 'llama3'."""
    model_map = {
        "phi3": "unsloth/Phi-3-medium-4k-instruct-bnb-4bit",
        "mistral": "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
        "llama3": "unsloth/llama-3-8b-bnb-4bit"
    }
    if model_name not in model_map:
        raise ValueError("model_name must be 'phi3', 'mistral', or 'llama3'")
    print(f"Loading {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_map[model_name],
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto"
    )
    return model, tokenizer

def clean_answer(raw_answer):
    """Post-process generated answer: remove extraneous text, ensure ends with period."""
    import re
    # Remove leading "Not found" if a correct answer follows
    if "Not found" in raw_answer and ("is" in raw_answer or "are" in raw_answer or "can be" in raw_answer):
        parts = re.split(r'Not found|### Solution \d+', raw_answer)
        if len(parts) > 1:
            raw_answer = parts[-1].strip()
    raw_answer = raw_answer.split("\n")[0].strip()
    if raw_answer and raw_answer[-1] not in ".!?":
        raw_answer += "."
    # Fix degree symbol encoding
    raw_answer = raw_answer.replace("Â°", "°")
    return raw_answer

def rag_answer(query, model, tokenizer, all_chunks, metadata, embed_model, index):
    """Generate answer for a single query using RAG."""
    chunks = retrieve_relevant_chunks(query, all_chunks, metadata, embed_model, index)
    context = "\n\n---\n\n".join([c["text"] for c in chunks])
    sources = list(set([c["recipe"] for c in chunks]))
    prompt = f"Text: {context}\n\nQuestion: {query}\n\nAnswer (one sentence):"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True,
        top_p=0.9,
        stop_strings=["?"],
        tokenizer=tokenizer
    )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer (one sentence):" in full:
        answer = full.split("Answer (one sentence):")[-1].strip()
    else:
        answer = full.strip()
    # Further cleanup
    answer = answer.split("Question:")[0].split("\n\n")[0].split("Source:")[0].strip()
    answer = clean_answer(answer)
    return {"answer": answer, "sources": sources}

def run_for_model(model_name):
    """Run the full pipeline for one model and save CSV."""
    # Build index (shared across models, but we rebuild each time for simplicity)
    all_chunks, metadata, embed_model, index = build_index(RECIPE_FOLDER)
    model, tokenizer = load_model(model_name)

    results = []
    for q in QUESTIONS:
        print(f"Processing: {q[:50]}...")
        try:
            resp = rag_answer(q, model, tokenizer, all_chunks, metadata, embed_model, index)
            results.append({
                "question": q,
                "answer": resp["answer"],
                "sources": ", ".join(resp["sources"])
            })
        except Exception as e:
            results.append({
                "question": q,
                "answer": f"ERROR: {str(e)}",
                "sources": ""
            })

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_FOLDER, f"rag_result_{model_name}.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Saved {csv_path}")

if __name__ == "__main__":
    # Example: run for Llama-3 (change to 'phi3' or 'mistral' and restart runtime)
    run_for_model("llama3")