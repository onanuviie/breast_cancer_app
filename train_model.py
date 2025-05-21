import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
import pickle
import os

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Pick the 10 features your app needs
selected_features = [
    'mean concavity', 'worst area', 'worst concave points', 'worst radius',
    'area error', 'worst concavity', 'mean concave points', 'worst symmetry',
    'radius error', 'worst texture'
]
X_selected = X[selected_features]

# Split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save
os.makedirs('model', exist_ok=True)
with open('model/logistic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully.")
