#Base64

import os
import numpy as np
import base64
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

app = Flask(__name__)

def predict_image(model, img_array):
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    accuracy = float(prediction[0][0]) if prediction[0][0] > 0.5 else float(1 - prediction[0][0])
    predicted_class = 'Dermatite' if prediction[0][0] > 0.5 else 'Normal'
    
    return predicted_class, accuracy

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({"error": "No image part"}), 400

    try:
        # Ensure proper base64 padding
        base64_image = data['image']
        missing_padding = len(base64_image) % 4
        if missing_padding:
            base64_image += '=' * (4 - missing_padding)
        
        # Decode the base64 image
        img_data = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        
        # Use the filename or a key from JSON to infer the true class if available
        file_name = data.get('filename', 'Unknown')
        if 'Lumpy_Skin' in file_name or 'dermatite' in file_name:
            true_class = 'Dermatite'
        elif 'Normal_Skin' in file_name or 'normal' in file_name:
            true_class = 'Normal'
        else:
            true_class = 'Unknown'

        predicted_class, accuracy = predict_image(model, img_array)
        
        # Determine if the prediction is positive or negative
        result = "positivo" if predicted_class == "Dermatite" else "negativo"
        
        return jsonify({
            "true_class": true_class,
            "predicted_class": predicted_class,
            "result": result,
            "accuracy": accuracy
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    model_path = 'bovino_dermatite_model_final.keras'
    if not os.path.exists(model_path):
        print("Modelo não encontrado. Treine o modelo primeiro.")
    else:
        model = load_model(model_path)
        app.run(debug=True)
