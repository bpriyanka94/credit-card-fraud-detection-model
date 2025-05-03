import gradio as gr
import joblib
import numpy as np
import json

# Load the trained model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define the prediction function to accept JSON input
def predict_fraud(input_text):
    try:
        input_json = json.loads(input_text)
        features = input_json.get("features", [])
        if len(features) != 29:
            return f"Error: Expected 29 features, got {len(features)}"
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)[0]
        return "Fraud Detected!!" if prediction == 1 else "Safe Transaction!!"
    except Exception as e:
        return f"Invalid input: {str(e)}"

# Launch the Gradio interface
demo = gr.Interface(
    fn=predict_fraud,
    inputs=gr.Textbox(label="Paste JSON here", lines=10, placeholder='{"features": [0.1, 0.2, ..., 149.62, 0.0]}'),
    outputs="text",
    title="Credit Card Fraud Detection",
    description="Paste the raw JSON input with a 'features' key containing 30 numeric values."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
