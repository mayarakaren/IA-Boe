import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify
import os
import argparse

app = Flask(__name__)

def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def create_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Congela a base pré-treinada

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def evaluate_model(model, test_generator):
    # Obter predições do modelo
    test_generator.reset()
    predictions = model.predict(test_generator, steps=test_generator.n // test_generator.batch_size + 1)
    predicted_classes = (predictions > 0.5).astype(int).reshape(-1)

    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Gerar a matriz de confusão
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Relatório de classificação
    print('Classification Report')
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)

def train_and_evaluate_model():
    set_seeds()

    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_dir = 'data/train'
    validation_dir = 'data/validation'
    test_dir = 'data/test'

    while True:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary'
        )

        validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary'
        )

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            shuffle=False  # Importante para garantir que as predições e rótulos correspondam corretamente
        )

        # Class weights to handle imbalance
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(train_generator.classes),
            y=train_generator.classes
        )
        class_weights = {i: class_weights[i] for i in range(len(class_weights))}

        checkpoint_cb = ModelCheckpoint('bovino_dermatite_model_best.keras', save_best_only=True, monitor='val_accuracy', mode='max')
        early_stopping_cb = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        cnn_model = create_model()

        history = cnn_model.fit(
            train_generator,
            epochs=15,
            validation_data=validation_generator,
            callbacks=[checkpoint_cb, early_stopping_cb],
            class_weight=class_weights
        )

        val_loss, val_accuracy = cnn_model.evaluate(validation_generator)
        print(f"Perda na validação: {val_loss}")
        print(f"Acurácia na validação: {val_accuracy}")

        if val_accuracy > 0.75:  # Suponha que 75% seja um bom ponto de referência
            cnn_model.save('bovino_dermatite_model_final.keras')

            test_loss, test_accuracy = cnn_model.evaluate(test_generator)
            print(f"Perda no teste: {test_loss}")
            print(f"Acurácia no teste: {test_accuracy}")

            # Avaliar modelo no conjunto de teste
            evaluate_model(cnn_model, test_generator)

def predict_image(model, img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    accuracy = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    predicted_class = 'Dermatite' if prediction[0][0] > 0.5 else 'Normal'
    
    # Imprime o caminho do arquivo e a predição
    print(f"Arquivo: {img_path}, Predição: {predicted_class}, Acurácia: {accuracy}")
    
    return predicted_class, accuracy

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = "temp.jpg"
        file.save(file_path)
        
        # Determina a classe verdadeira com base no caminho do arquivo (substitua isso conforme necessário)
        true_class = 'Unknown'
        if 'lumpy_skin' in file.filename.lower():
            true_class = 'Dermatite'
        elif 'normal_skin' in file.filename.lower():
            true_class = 'Normal'
        
        class_name, accuracy = predict_image(model, file_path)
        os.remove(file_path)  # Remove o arquivo temporário
        
        # Imprime a classe verdadeira para depuração
        print(f"Classe verdadeira: {true_class}")
        
        return jsonify({"true_class": true_class, "predicted_class": class_name, "accuracy": accuracy}), 200

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model or run Flask app for predictions.')
    parser.add_argument('--train', action='store_true', help='Train the model')
    args = parser.parse_args()

    if args.train:
        # Treinar e salvar o modelo
        train_and_evaluate_model()
    else:
        # Carregar o modelo e executar a API
        if not os.path.exists('bovino_dermatite_model_final.keras'):
            print("Modelo não encontrado. Execute com --train para treinar o modelo.")
        else:
            model = load_model('bovino_dermatite_model_final.keras')
            app.run(debug=True)
