from flask import Flask, request, jsonify
import pickle
import numpy as np

# Create a Flask application instance
app = Flask(__name__)

# Load the pickled model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define a route for the home page
@app.route("/")
def home():
    return "Welcome to the Flask ML App!"

# Define a route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse input JSON
        input_data = request.json["data"]  # Expect JSON like {"data": [1.0, 2.0, 3.0]}
        input_array = np.array(input_data).reshape(1, -1)  # Reshape for model
        
        # Predict using the model
        prediction = model.predict(input_array)
        
        # Return the prediction
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app (only for local development)
if __name__ == "__main__":
    app.run(debug=True)
