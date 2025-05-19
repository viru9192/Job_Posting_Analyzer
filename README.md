# 🕵️‍♂️ Job Posting Analyzer — Fake Job Detection using NLP + ML

This project is a complete machine learning pipeline for detecting **fake job postings** using **Natural Language Processing** (TF-IDF), **feature engineering**, and **XGBoost** classifier. It includes a user-friendly **Streamlit web interface** for real-time predictions.

---

## 📌 Project Overview

With the rise of online job scams, it's crucial to automatically detect fraudulent postings. Our system is trained on real-world data and uses a hybrid of textual and structured features for classification.

---

## 🧠 Features

- **NLP Feature Extraction**:
  - TF-IDF vectorization (bi-grams, 7000 features)
  - Suspicious keyword detection (e.g., "urgent hire", "investment opportunity")

- **Engineered Features**:
  - `avg_salary`, `is_remote_flag`, `suspicious_keyword_count`, `years_experience`

- **ML Pipeline**:
  - Class imbalance handled via **SMOTE**
  - Models compared: Logistic Regression, Random Forest, and XGBoost
  - **XGBoost** selected as best model (~98.85% accuracy)

- **Frontend**:
  - Built using **Streamlit**
  - Takes job description input and shows prediction with confidence score

---

## 🖥️ Web App Screenshots

### 🔺 Fake Job Example
![Fake Job Example](Job_Posting_Fake.jpg)

---

### ✅ Real Job Example
![Real Job Example](Job_Posting_Real.jpg)

---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/viru9192/Job_Posting_Analyzer.git
cd Job_Posting_Analyzer
