# MachineLearning

This repository contains a collection of machine learning exercises, experiments, and utilities, organized by topic and project. Each folder represents a unique part of the coursework or a specific machine learning task.

## Repository Structure

### Sheet02

- **Purpose:** Implements and compares K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) algorithms for classifying satellite images from the [EuroSAT](https://github.com/phelber/EuroSAT) dataset.
- **Contents:**
  - `02_knn_svm.py`: Main script for running KNN and SVM experiments.

### Sheet04

- **Purpose:** Applies the Expectation-Maximization (EM) algorithm to categorize 2D point clouds and extract keyframes from high-dimensional feature vectors.
- **Contents:**
  - `em.py`: EM algorithm implementation.
  - `misc.py`: Helper functions and utilities.
  - `test.py`: Test scripts for the algorithms.
  - `vids/`: Example video files for processing.

### Sheet06

- **Purpose:** Explores natural language processing using word embeddings and recipe data with word2vec.
- **Contents:**
  - `voc.py`, `w2v.py`: Scripts for vocabulary processing and word2vec embedding.

### Sheet07

- **Purpose:** Assigns categories to recipes using BERT-based text classification.
- **Contents:**
  - `berttest.py`: Script for testing BERT-based classification on recipe data.
  - `template/`: Contains the code fot assigning categories to recipes:
    - `config.py`: Configuration file for model and training parameters.
    - `dataset.py`: Defines the dataset class and utilities for loading and processing recipe data.
    - `model.py`: Contains the RecipERT model architecture, a multi-label classifier based on a transformer backbone.
    - `test.py`: Script for evaluating the trained model on test data and reporting precision, recall, and F1-score.
    - `train.py`: Script for training the RecipERT model, including data loading, batching, and checkpointing.

### Sheet08

- **Purpose:** Investigates the ability to determine which large language model (LLM) generated a given text sample, focusing on LLM attribution and analysis.
- **Contents:**
  - `config.py`: Configuration file for setting up experiments and specifying model parameters.
  - `which_lm.py`: Main script for analyzing and predicting which LLM produced a given text.

---
