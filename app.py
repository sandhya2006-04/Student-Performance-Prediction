from flask import Flask, render_template, request, jsonify
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import pickle

app = Flask(__name__)

# Training data
# Features:
# study_hours, attendance_percent, assignment_marks,
# internal_marks, previous_marks, sleep_hours

X = np.array([
    [2, 60, 10, 10, 50, 5],
    [3, 65, 12, 12, 55, 6],
    [4, 70, 14, 14, 60, 6],
    [5, 75, 16, 16, 65, 7],
    [6, 80, 18, 18, 70, 7],
    [7, 85, 20, 20, 75, 8],
    [8, 90, 22, 22, 80, 8],
    [9, 95, 24, 24, 85, 9]
])

y = np.array([50, 55, 60, 65, 70, 75, 82, 90])

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Save fitted model
with open("student_model.pkl", "wb") as file:
    pickle.dump(model, file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    study_hours = float(data["study_hours"])
    attendance = float(data["attendance_percent"])
    assignment = float(data["assignment_marks"])
    internal = float(data["internal_marks"])
    previous = float(data["previous_marks"])
    sleep = float(data["sleep_hours"])

    features = np.array([[
        study_hours,
        attendance,
        assignment,
        internal,
        previous,
        sleep
    ]])

    prediction = model.predict(features)[0]

    # Keep result between 0 and 100
    prediction = max(0, min(100, prediction))

    return jsonify({
        "predicted_marks": round(prediction, 2),
        "remarks": "Performance prediction completed successfully."
    })


if __name__ == "__main__":
    app.run(debug=True)