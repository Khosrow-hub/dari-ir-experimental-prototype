# Dari Information Retrieval Experimental Prototype

An experimental Dari Information Retrieval (IR) system developed to investigate the effectiveness of multiple retrieval techniques in a low-resource language environment.

The project evaluates and compares Exact Matching, Fuzzy Matching, Character Trigram Similarity, TF-IDF Retrieval, Hybrid Retrieval, and Query Expansion approaches using a custom Dari corpus and benchmark query dataset.

## Research Objective

The objective of this project is to explore methods for improving text retrieval robustness in Dari, particularly when user queries contain:

* Typographical errors
* Incomplete search expressions
* Lexical variation
* Orthographic inconsistencies

The system was developed as part of an academic research study in the field of Information Retrieval.

## Features

* Dari text normalization
* Exact keyword matching
* Fuzzy Matching using RapidFuzz
* Character Trigram Similarity Retrieval
* TF-IDF Retrieval
* Hybrid Retrieval Model
* Query Expansion
* Top-K Retrieval Ranking
* Retrieval Evaluation Framework
* Statistical Performance Comparison

## Dataset

The experimental corpus contains:

* 514 Dari text records

The evaluation benchmark contains:

* 150 manually curated Dari search queries

Query categories:

* Exact Queries
* Typographical Error Queries
* Partial Queries
* Lexical Variant Queries

## Evaluation Metrics

The retrieval models were evaluated using:

* Top-5 Accuracy
* Precision@5
* Recall@5
* F1-Score
* Query-Type Performance Analysis
* McNemar Statistical Significance Test

## Retrieval Models

### Baseline Methods

1. Exact Matching
2. Fuzzy Matching
3. Character Trigram Similarity
4. TF-IDF Retrieval

### Proposed Method

Expanded Hybrid Retrieval Model:

* Query Normalization
* Query Expansion
* Fuzzy Similarity Scoring
* Character Trigram Similarity Scoring
* Hybrid Ranking

## Technologies Used

* Python
* Streamlit
* Pandas
* Scikit-learn
* RapidFuzz
* NumPy
* SciPy

## Project Structure

```text
essaySystem/
│
├── app.py
├── dari_clean_corpus.csv
├── dari_evaluation_queries_150.csv
├── requirements.txt
├── README.md
└── results/
```

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Research Significance

Dari is considered a low-resource language with limited publicly available information retrieval benchmarks and linguistic resources. This project investigates practical retrieval strategies that improve robustness against spelling variation and query inconsistency without requiring large-scale neural models or extensive annotated datasets.

## Academic Purpose

This repository contains an experimental research prototype developed for academic and educational purposes in the field of Information Retrieval and Natural Language Processing.

## License

This project is intended for educational and research use.
