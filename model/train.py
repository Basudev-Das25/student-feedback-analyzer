import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import os
import pickle


#Load the dataset
data = pd.read_csv("../data/finalDataset0.2.csv")

#keep only required columns
data = data[["teaching.1", "teaching"]]

#Drop missing values
data = data.dropna()

#labelEncoding
label_encoder = LabelEncoder()
data["sentiment_encoded"] = label_encoder.fit_transform(data["teaching"])


X_text = data["teaching.1"]
y = data["sentiment_encoded"]

#Text Vectorization(TF- IDF)
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(X_text)

#train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#Logistic Regression model
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

#Prediction
y_pred = model.predict(X_test)


print("\nClassification Report")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
    )
)

# ===== STEP 9: NLP EXPLAINABILITY =====

# Get feature (word) names from TF-IDF
# feature_names = vectorizer.get_feature_names_out()

# Coefficients: shape (n_classes, n_features)
# coefficients = model.coef_

# print("\nCoefficient matrix shape:", coefficients.shape)

# def show_top_words(class_index, top_n=10):
#     class_coef = coefficients[class_index]

#     top_positive_idx = np.argsort(class_coef)[-top_n:][::-1]
#     top_negative_idx = np.argsort(class_coef)[:top_n]

#     print(f"\nTop positive words for class {class_index}:")
#     for idx in top_positive_idx:
#         print(f"  {feature_names[idx]} : {class_coef[idx]:.3f}")

#     print(f"\nTop negative words for class {class_index}:")
#     for idx in top_negative_idx:
#         print(f"  {feature_names[idx]} : {class_coef[idx]:.3f}")

# # Display top words for each class
# for class_idx in range(len(model.classes_)):
#     show_top_words(class_idx)

#Absolute path to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#App directory
APP_DIR = os.path.join(BASE_DIR, "app")
os.makedirs(APP_DIR, exist_ok=True)

#Save trained model
with open(os.path.join(APP_DIR, "sentiment_model.pkl"), "wb") as f:
    pickle.dump(model, f)

#Save TF-IDF vectorizer
with open(os.path.join(APP_DIR, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

#Save label_encoder
with open(os.path.join(APP_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

print("\nModel, vectorizer, and label_encoder saved successfully")