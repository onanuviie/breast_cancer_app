from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join("model", "log_reg_top10_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form
        features = [
            float(request.form['worst_texture']),
            float(request.form['radius_error']),
            float(request.form['worst_symmetry']),
            float(request.form['mean_concave_points']),
            float(request.form['worst_concavity']),
            float(request.form['area_error']),
            float(request.form['worst_radius']),
            float(request.form['worst_concave_points']),
            float(request.form['worst_area']),
            float(request.form['mean_concavity'])
        ]
        features = np.array(features).reshape(1, -1)

        prediction = model.predict(features)[0]
        label = "Malignant" if prediction == 1 else "Benign"
        return render_template('result.html', result=label)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
