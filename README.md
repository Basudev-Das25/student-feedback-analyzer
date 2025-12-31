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
