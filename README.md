# Medical RAG Pipeline: Open-Source & Custom-Built 🩺

## Overview

This repository contains a 100% custom-built Retrieval-Augmented Generation (RAG) pipeline designed specifically for extracting insights and querying complex medical texts, specifically targeting hypertension treatment protocols. It relies exclusively on open-source, locally hosted models—including a medically fine-tuned LLM—ensuring full data privacy and superior retrieval accuracy.

## Pipeline Architecture

The system is engineered from scratch and follows a rigorous, multi-stage pipeline:

* **Data Ingestion & Preparation:** Transforms unstructured raw data (e.g., OCR JSON) into a usable format through dynamic environment setup and robust file operations.
* **Text Cleaning & Structuring:** Deeply cleans raw text by stripping HTML, handling structural noise, removing boilerplate, and detecting section boundaries.
* **Intelligent Chunking:** Divides structured text into manageable, section-aware chunks that preserve critical medical context and overlap seamlessly.
* **Metadata Generation & Validation:** Enriches each chunk with canonical metadata and systematically validates the dataset for maximum integrity.
* **Embedding & Indexing:** Converts text chunks into numerical vectors using a specialized medical embedding model and indexes them via FAISS for high-speed similarity search.
* **Retrieval:** Queries the vector index to extract the most highly relevant document chunks using dense semantic matching.
* **Answer Generation:** Synthesizes a highly grounded, citation-backed response using a locally hosted Language Model (LLM).
* **Safety & Refusal Layer:** Acts as a strict guardrail by assessing retrieval confidence—refusing to answer or issuing a caution if the retrieved evidence is insufficient.
* **Evaluation & Demonstration:** Systematically tests the end-to-end pipeline, evaluating retrieval quality and answer generation through a custom heuristic metric.

## Models Utilized

* **Embedding Model:** `sentence-transformers/embeddinggemma-300m-medical` (with `all-MiniLM-L6-v2` as a robust fallback mechanism).
* **Large Language Model (LLM):** `Qwen/Qwen2.5-1.5B-Instruct` (a highly efficient, medically fine-tuned model hosted entirely locally for secure, grounded generation).

## Tech Stack & Key Libraries

* **Core Infrastructure:** `subprocess`, `sys`, `os`, `warnings`
* **Data Processing & Cleaning:** `re`, `json`, `html`, `unicodedata`, `collections.Counter`
* **Machine Learning & Math:** `numpy`, `torch`
* **NLP & Embeddings:** `sentence_transformers`, `transformers`
* **Vector Database:** `faiss`
