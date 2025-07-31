# Sentiment Analysis Repository

## Overview

Welcome to the **Sentiment Analysis** repository! This repository is dedicated to exploring and practicing sentiment analysis techniques on various datasets. The goal is to build, evaluate, and compare machine learning models for classifying text data into sentiment categories (e.g., Positive, Negative, Neutral). Each project focuses on a specific dataset, applying natural language processing (NLP) techniques and machine learning algorithms to analyze sentiment.

This repository serves as a portfolio of sentiment analysis projects, showcasing different approaches, preprocessing techniques, models, and evaluation metrics. It is designed to be a learning resource for experimenting with text classification and improving NLP skills.

## Projects

Below is a list of sentiment analysis projects completed in this repository, with details on the dataset, methodology, and outcomes.

### 1. Amazon Reviews Sentiment Analysis

- **Dataset**: Amazon Fine Food Reviews dataset, available on Kaggle: [Amazon Product Reviews](https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews)
- **Description**: This project analyzes customer reviews from the Amazon Fine Food Reviews dataset to classify sentiments as `Positive`, `Neutral`, or `Negative`. The dataset contains approximately 568,454 reviews with ratings (1–5), where scores are mapped to sentiments (1–2: Negative, 3: Neutral, 4–5: Positive). The dataset exhibits significant class imbalance (~78% Positive, 7.5% Neutral, 14.4% Negative).
- **Preprocessing**:
  - Cleaned missing values using `dropna` for `Score` and filled missing `Text` with spaces.
  - Used `StratifiedShuffleSplit` to create training (454,763 reviews) and test (113,691 reviews) sets, preserving class distribution.
  - Applied `CountVectorizer` for bag-of-words representation and `TfidfTransformer` for TF-IDF weighting.
- **Models**:
  - **Multinomial Naive Bayes**: Achieved 79.77% accuracy but struggled with `Neutral` (F1: 0.00) and `Negative` (F1: 0.22) due to class imbalance.
  - **Logistic Regression**: Improved performance with 87.77% accuracy, with better F1-scores for `Negative` (0.74) and `Neutral` (0.34).
- **Evaluation**:
  - Metrics: Accuracy, precision, recall, F1-score, and confusion matrix.
  - Custom metrics for `Positive` class were initially incorrect (omitted `Neutral` reviews in calculations) but corrected to align with scikit-learn’s `classification_report`.
  - Visualizations: Histograms, box plots, violin plots, and scatter plots explored data distribution; confusion matrices visualized model performance.
- **Files**:
  - `amazon_reviews_sentiment_analysis.ipynb`: Jupyter notebook with data preprocessing, EDA, modeling, and evaluation.
  - `data/`: Directory for the dataset (not included in the repo due to size; download from [Kaggle](https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews)).
- **Future Improvements**:
  - Implement text preprocessing (e.g., stop word removal, lemmatization).
  - Add class weighting or oversampling to address imbalance.
  - Experiment with advanced models (e.g., Random Forest, deep learning with BERT).

## Future Projects

This repository will be updated with additional sentiment analysis projects on different datasets, such as:

- Twitter sentiment analysis
- IMDb movie reviews
- Yelp business reviews
  Each project will follow a similar structure: data exploration, preprocessing, model training, evaluation, and visualization.

## Getting Started

To run the projects in this repository:

1. Clone the repository:
   ```bash
   git clone https://github.com/1mustyz/sentiment-analysis.git
   ```
