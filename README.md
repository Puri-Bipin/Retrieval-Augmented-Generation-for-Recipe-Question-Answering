# RAG Recipe Assistant: Comparing Phi-3, Mistral-7B, and Llama-3

This repository contains the code and results for a **Retrieval-Augmented Generation (RAG)** system that answers questions about cooking recipes. Three open-source LLMs — **Microsoft Phi-3 Medium**, **Mistral-7B Instruct**, and **Meta Llama-3 8B** — are compared on 13 domain-specific questions.

---

## Features

- Chunk and index recipe text using `all-MiniLM-L6-v2` embeddings and FAISS.
- Retrieve the top-*k* most relevant chunks for a user question.
- Generate a one-sentence answer using any of the three LLMs (4-bit quantized).
- Evaluate answers on:
  - **Faithfulness**
  - **Answer Relevance**
  - **Retrieval Precision**
  - **Recall**
- Produce CSV outputs for quantitative comparison.

---

## Requirements

- Google Colab (recommended) or a machine with at least **16GB GPU RAM** (T4 or better)
- Python **3.10+**
- Libraries listed in `requirements.txt`

---

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/Puri-Bipin/Retrieval-Augmented-Generation-for-Recipe-Question-Answering.git
cd recipe-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place your recipe `.txt` files in:

```bash
data/recipes/
```

Example recipe files are provided.

---

## Usage

## Run the RAG pipeline for one model

The script `code/rag_pipeline.py` is self-contained. Modify the `model_name` inside the `if __name__ == "__main__":` block, then run:

```bash
python code/rag_pipeline.py
```

By default, it runs for `llama3`.

To run for Phi-3 or Mistral, change the line to:

```python
run_for_model("phi3")   # or "mistral"
```

> **Important:** Because the models are large (7–14B parameters), restart your Python kernel or terminal between runs to free GPU memory. On Google Colab, simply restart the runtime.

---

## What the script does

- Load recipe `.txt` files from `data/recipes/`
- Split them into chunks of **800 characters** with **100-character overlap**
- Build a FAISS index using `all-MiniLM-L6-v2` embeddings
- For each of the **13 predefined questions**, retrieve the **top 6 chunks**
- Generate a one-sentence answer using the selected LLM
- Save the results as:

```bash
data/results/rag_result_{model_name}.csv
```

---

## Evaluate all three models

After generating CSV files for all three models:

- `rag_result_phi3.csv`
- `rag_result_mistral.csv`
- `rag_result_llama3.csv`

place them inside:

```bash
data/results/
```

Then run:

```bash
python code/evaluation.py
```

---

## What the evaluation script does

- Rebuild the same FAISS index from the recipe texts
- Load the three CSV files (handling encoding issues automatically)
- Compute:
  - **Faithfulness**
  - **Answer Relevance**
  - **Retrieval Precision**
  - **Recall**
- Print average metrics to the console
- Save two files inside `data/results/`:

```bash
detailed_metrics_all_models.csv
summary_metrics.csv
```

Where:

- `detailed_metrics_all_models.csv` = per-question scores for every model
- `summary_metrics.csv` = average metrics per model

---

## Run custom questions (Advanced)

If you want to use your own questions, open `code/rag_pipeline.py` and replace the `QUESTIONS` list at the top.

---

## Results Summary

From our experiments on **13 recipe-based questions**:

| Model   | Faithfulness | Answer Relevance | Retrieval Precision | Recall |
|---------|--------------|------------------|---------------------|--------|
| Phi-3   | 0.757        | 0.828            | 0.420               | 1.000  |
| Mistral | 0.770        | 0.807            | 0.423               | 1.000  |
| Llama-3 | 0.792        | 0.761            | 0.419               | 1.000  |

---

## Observations

- **Phi-3** gives the most natural and complete answers, making it the best choice for a user-facing assistant.
- **Llama-3** is the most faithful to the retrieved text, but is often too concise.
- **Mistral** performs well overall, but made one factual error regarding the protein content of roast chicken.

---

## Project Structure

```bash
recipe-rag/
│
├── code/
│   ├── rag_pipeline.py
│   └── evaluation.py
│
├── data/
│   ├── recipes/
│   └── results/
│
├── requirements.txt
└── README.md
```

---

## Notes

- This pipeline is designed for recipe question answering, but it can easily be adapted to other domain-specific document collections.
- Model performance may vary depending on GPU, document quality, and question phrasing.

---

## Citation / Reference

If you use this project in academic work, please cite the corresponding report or repository.

---

## License

This project is for educational and research purposes.