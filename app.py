from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load your trained model
model = load_model('models/sigmoid_model_2.h5')

# Load your tokenizer here
tokenizer = Tokenizer()  # Modify this line as necessary to load your tokenizer

def get_prediction(text):
    # Preprocess the text so it matches the training data
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=100)  # Modify maxlen as per your model's input shape

    # Get the model's prediction
    prediction = model.predict(np.array(padded_sequence))

    # Translate the model's prediction to a sentiment ('positive', 'negative', etc.)
    sentiment = 'positive' if prediction > 0.5 else 'negative'  # Modify this line as per your model's output interpretation

    return sentiment

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text")

    # Process the text and get a prediction from your model
    sentiment = get_prediction(text)
    prediction_text = f"The predicted sentiment is: {sentiment}"

    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
