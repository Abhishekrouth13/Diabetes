from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open('diabetes_model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None

    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        data = np.array([features])
        prediction = model.predict(data)[0]
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        prediction_text = f'Prediction: {result}'

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
