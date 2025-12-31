from flask import Flask, render_template, request
import os
import pickle

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

#Resolve paths relative to this file
APP_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(APP_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(APP_DIR, "vectorizer.pkl")
ENCODER_PATH = os.path.join(APP_DIR, "label_encoder.pkl")

#Load
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def home():
    result_text = "Enter feedback text and click Analyze"

    if request.method == "POST":
        text = request.form.get("feedback", "").strip()

        if text:
            X = vectorizer.transform([text])
            pred = model.predict(X)[0]

            sentiment_map = {
                0: "Neagtive",
                1: "Neutral",
                2: "Positive"
            }

            label = sentiment_map.get(int(pred), "Unkown")

            result_text = f"Predicted Sentiment: {label}"

        else:
            result_text = "Please enter some feedback text."

    return render_template("index.html", result_text = result_text)

if __name__ == "__main__":
    app.run()