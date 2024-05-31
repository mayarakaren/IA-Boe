import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)

def predict_image(model, img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    accuracy = float(prediction[0][0]) if prediction[0][0] > 0.5 else float(1 - prediction[0][0])
    predicted_class = 'Dermatite' if prediction[0][0] > 0.5 else 'Normal'
    
    # Infer true class based on file name
    file_name = os.path.basename(img_path)
    if 'Lumpy_Skin' in file_name or 'dermatite' in file_name:
        true_class = 'Dermatite'
    elif 'Normal_Skin' in file_name or 'normal' in file_name:
        true_class = 'Normal'
    else:
        true_class = 'Unknown'

    return true_class, predicted_class, accuracy

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = "temp_image.jpg"
        file.save(file_path)
        true_class, predicted_class, accuracy = predict_image(model, file_path)
        os.remove(file_path)
        
        # Determine if the prediction is positive or negative
        result = "positivo" if predicted_class == "Dermatite" else "negativo"
        
        return jsonify({
            "predicted_class": predicted_class,
            "result": result,
            "accuracy": accuracy
        }), 200

if __name__ == '__main__':
    model_path = 'bovino_dermatite_model_final.keras'
    if not os.path.exists(model_path):
        print("Modelo não encontrado. Treine o modelo primeiro.")
    else:
        model = load_model(model_path)
        app.run(debug=True)
