# RAG Recipe Assistant: Comparing Phi‑3, Mistral, Llama‑3

This repository contains the code and results for a **Retrieval-Augmented Generation (RAG)** system that answers questions about cooking recipes. Three open‑source LLMs (Microsoft Phi‑3 Medium, Mistral‑7B Instruct, Meta Llama‑3 8B) are compared on 13 domain‑specific questions.

## Features

- Chunk and index recipe text using `all-MiniLM-L6-v2` embeddings and FAISS.
- Retrieve top‑k relevant chunks for a user question.
- Generate a one‑sentence answer using any of the three LLMs (4‑bit quantized).
- Evaluate answers on **faithfulness**, **answer relevance**, **retrieval precision**, and **recall**.
- Produce CSV outputs for quantitative comparison.

## Requirements

- Google Colab (recommended) or a machine with at least 16GB GPU RAM (T4 or better).
- Python 3.10+
- Libraries listed in `requirements.txt`

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/Puri-Bipin/Retrieval-Augmented-Generation-for-Recipe-Question-Answering.git
   cd recipe-rag