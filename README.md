# 🐦 Twitter Sentiment Analysis

This repo contains two approaches to sentiment analysis on tweets:

1. **Traditional ML (TF-IDF + Logistic Regression)**
2. **Transformer-based Model (RoBERTa)**

---

## 📌 Project 1: ML-Based Sentiment Classifier

- Uses text preprocessing, TF-IDF, and Logistic Regression.
- Outputs binary sentiment: Positive or Negative.

**Steps:**
- Clean tweets using regex.
- Vectorize with TF-IDF.
- Train/test using `train_test_split`.
- Predict sentiment on new input.

📁 File: `twittersentiment(1).ipynb`  
🧪 Input: Raw tweets  
📤 Output: 0 (Negative), 1 (Positive)

---

## 🤖 Project 2: RoBERTa Sentiment Classifier

- Uses HuggingFace `transformers` and `pipeline`.
- Model: `roberta-base`
- GPU-compatible for fast predictions.

**Steps:**
- Clean text.
- Load tokenizer & model.
- Predict using sentiment-analysis pipeline.
- Map labels to binary sentiment.

📁 File: `twittersentimentanalysis-roberta.ipynb`  
📤 Output: 0 (Negative), 1 (Positive)

---

## 📦 Requirements

```bash
pip install pandas, scikit-learn, nltk, matplotlib, seaborn, torch, transformers
