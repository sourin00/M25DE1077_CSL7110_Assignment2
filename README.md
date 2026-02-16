# CSL7110 -- Assignment 2
## MinHash and Locality-Sensitive Hashing



## Overview

This project implements:

-   k-gram based Jaccard similarity computation
-   MinHash signature approximation
-   Locality Sensitive Hashing (LSH)
-   Large-scale similarity detection on the MovieLens 100k dataset

The goal is to analyze similarity detection techniques and study
their trade-offs in accuracy and efficiency.


## Project Structure

```
.
├── main_docs.py              # Q1–3 (D1–D4 documents)
├── movielens_minhash.py      # Q4 (MinHash on MovieLens)
├── movielens_lsh.py          # Q5 (LSH on MovieLens)
├── minhash/                  # Data for Q1,2,3 (D1.txt, D2.txt, D3.txt, D4.txt)
├── data/ml-100k/u.data       # MovieLens dataset
├── results/                  # Generated result files
└── README.md

```



## How to Run

### 1️. Install dependencies

pip install numpy

### 2️. Run document experiments (Parts 1--3)

python main_docs.py

### 3️. Run MinHash on MovieLens (Part 4)

python movielens_minhash.py

### 4️. Run LSH on MovieLens (Part 5)

python movielens_lsh.py

All outputs will be generated inside the `results/` directory.



## Key Findings

-   Increasing the number of hash functions reduces variance in MinHash
    estimates.
-   False positives decrease significantly as hash functions increase.
-   Proper tuning of LSH parameters (r, b) is essential for balancing
    false positives and false negatives.
-   LSH performs exceptionally well for high similarity thresholds
    (e.g., 0.8).


## Concepts Covered

-   Jaccard Similarity
-   MinHash approximation
-   Locality Sensitive Hashing (LSH)
-   Scalable similarity search


