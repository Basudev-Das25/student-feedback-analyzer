# Student Feedback Sentiment Analyzer (NLP)

An end-to-end **Natural Language Processing (NLP)** project that analyzes student feedback text and classifies sentiment as **Positive**, **Neutral**, or **Negative** using TF-IDF and Logistic Regression.

This project demonstrates the complete NLP pipeline: data preprocessing, feature extraction, supervised text classification, explainability, and deployment-ready inference.

---

## Features

- NLP-based sentiment classification on real student feedback
- TF-IDF vectorization for text feature extraction
- Logistic Regression with class balancing
- Explainable AI using feature coefficient analysis
- Flask-based inference web application
- Deployment-ready project structure

---

## Tech Stack

- Python
- pandas
- scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Flask
- HTML / CSS
- Gunicorn (deployment-ready)

---

## Project Structure

student-feedback-analyzer/
├── app/
│ ├── app.py
│ ├── sentiment_model.pkl
│ ├── vectorizer.pkl
│ ├── label_encoder.pkl
│ ├── templates/
│ │ └── index.html
│ └── static/
│ └── style.css
├── data/
│ └── finalDataset0.2.csv
├── model/
│ └── train.py
├── requirements.txt
└── README.md


---

##  Run Locally

in bash
-------
git clone https://github.com/YOUR_USERNAME/student-feedback-analyzer.git
cd student-feedback-analyzer
pip install -r requirements.txt
cd app
python app.py
Open in browser:http://127.0.0.1:5000

------

## How It Works

1. User enters student feedback text

2. Text is transformed using the saved TF-IDF vectorizer

3. Logistic Regression model predicts sentiment class

4. Numeric class is mapped to:

    0 → Negative
    1 → Neutral
    2 → Positive

5. Result is displayed in the UI

## Model Explainability

The model’s predictions are interpretable by analyzing TF-IDF feature coefficients.
Key words contributing to each sentiment class were extracted to validate model reasoning.

## Deployment (Optional)
This project is deployment-ready and can be hosted on platforms like Render.

Build Command
 bash:
 pip install -r requirements.txt

Start Command
 bash:
 gunicorn app.app:app

## Notes

1. Training and inference are fully separated

2. Model artifacts are versioned for reproducibility

3. Designed for clone-and-deploy usage

---

## ⚠️ Model Limitations & Design Decisions

This sentiment analyzer uses a TF-IDF + Logistic Regression pipeline, which provides fast, interpretable, and deployment-friendly NLP inference.

### Known Limitations
- Bag-of-words models do not fully capture semantic context or negation.
- Phrases like "not good" or "very bad" may be misclassified in rare cases.
- Minority sentiment classes have limited samples in the dataset, affecting recall.

### Mitigations Applied
- Class-weighted Logistic Regression to address imbalance.
- Bigram features (1–2 grams) to improve handling of negation and sentiment phrases.

### Future Improvements
- Utilize all textual feedback columns by combining them into a unified input.
- Explore hybrid models combining text and structured features.
- Evaluate transformer-based models (e.g., BERT) for deeper semantic understanding.


# Experimental Branch: Full-Text Combination

This repository includes an experimental branch that explores improving sentiment prediction by utilizing all available textual feedback fields in the dataset.

Branch Name:
    feature/full-text-combination

### Motivation

The baseline model was trained using a single high-signal feedback column to establish a clean and interpretable NLP pipeline. However, the dataset contains multiple complementary text fields (e.g., teaching, course content, lab work, extracurricular feedback), which provide additional context about student experience.

To better leverage this information, an experimental branch was created to combine all textual inputs into a unified document for model training and inference.

### What Was Changed
1. Combined multiple text columns into a single combined_text feature

2. Retained the same sentiment target (teaching) to avoid label ambiguity

3. Reused the same NLP pipeline:
    TF-IDF vectorization with unigram + bigram features
    Class-weighted Logistic Regression

4. Updated the Flask inference app to accept multiple feedback inputs and combine them consistently with training

## Outcome
1. Improved contextual understanding of feedback

2. Better handling of mixed sentiment statements

3. More realistic behavior for negative and neutral feedback cases

The baseline model remains available on the main branch for simplicity and stability, while this branch serves as a documented enhancement and experimentation path.

## Engineering Rationale
This branching approach reflects real-world ML development practices:
1. Stable baseline maintained on main
2. Experimental improvements isolated in feature branches
3. Trade-offs documented rather than hidden

--------

> This project follows an iterative ML development approach, balancing deployable baselines with documented experimentation using Git branching.
