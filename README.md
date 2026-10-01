# 🛡️ Fake Job Posting Detection

A machine learning project that classifies job postings as **Real** or
**Fake** using NLP and Logistic Regression, deployed as an interactive
Streamlit web app.

---

## 📁 Project Structure

```
fake-job-posting-detection/
├── app.py                          # Streamlit web app for real-time prediction
├── requirements.txt                # Python dependencies
├── data/
│   └── Fake_Postings.csv           # Dataset used for training
├── model/
│   ├── fake_job_model.pkl          # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl        # Fitted TF-IDF vectorizer
├── notebook/
│   └── fake_job_posting.ipynb      # Full EDA, preprocessing & model training
└── README.md
```

---

## 🎯 Problem Statement

Online job platforms are frequently exploited to post fraudulent listings
that steal personal information or money from applicants. This project
builds a text-classification model to flag suspicious job postings
automatically.

---

## 🧪 Approach

1. **Data Preprocessing** — handled missing values, removed duplicates,
   cleaned text (lowercasing, punctuation removal), balanced classes to
   reflect a realistic real-vs-fake distribution.
2. **Feature Engineering** — TF-IDF vectorization on job description text
   (`max_features=3000`, unigrams + bigrams, `min_df=2`).
3. **Modeling** — Logistic Regression with `class_weight='balanced'` and
   `C=0.1` regularization to prevent overfitting on the imbalanced data.
4. **Evaluation** — accuracy, ROC-AUC, precision/recall/F1 per class.
5. **Deployment** — Streamlit app that loads the saved model + vectorizer
   and predicts on user-entered job text in real time.

---

## 📊 Results

| Metric | Score |
|---|---|
| Accuracy | **87.85%** |
| ROC-AUC | **0.8747** |
| Precision (Fake) | 0.56 |
| Recall (Fake) | 0.85 |
| F1-score (Fake) | 0.68 |

> Recall on the "Fake" class is prioritized — in fraud detection, missing a
> fake posting (false negative) is costlier than a false alarm.

---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repo and move into the project folder
cd fake-job-posting-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run app.py
```

To retrain the model from scratch, open `notebook/fake_job_posting.ipynb`
in Jupyter and run all cells — it will regenerate `fake_job_model.pkl` and
`tfidf_vectorizer.pkl`.

---

## 🛠️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · TF-IDF · Logistic Regression ·
Matplotlib · Seaborn · Streamlit · Joblib

---

## 🚀 Possible Extensions

- Try ensemble models (Random Forest, XGBoost) for comparison
- Add SHAP/LIME explainability to show which words drove a prediction
- Deploy on Streamlit Community Cloud for a live public demo link

---

**Author:** Iqra Siddiqui
