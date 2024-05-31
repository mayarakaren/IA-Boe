# Classificação de Imagens com Inteligência Artificial - Dermatite Nodular Contagiosa

Este repositório contém um projeto de aprendizado profundo utilizando TensorFlow e Keras para classificar imagens de bovinos, determinando se apresentam ou não dermatite nodular contagiosa. O modelo utiliza Redes Neurais Convolucionais (CNN) e uma Perceptron de Múltiplas Camadas (MLP). As CNNs são eficazes no processamento de dados visuais por meio de suas camadas convolucionais que detectam características importantes nas imagens, enquanto as MLPs são redes neurais feedforward compostas por múltiplas camadas que podem modelar relações complexas nos dados.

![Imagem do Projeto](github/slide01.png)


## Índice

1. [Instalação](#instalação)
2. [Descrição do Código](#descrição-do-código)
3. [Treinamento do Modelo](#treinamento-do-modelo)
4. [API com Flask](#api-com-flask)
5. [Uso](#uso)
6. [Resultados](#resultados)
7. [Observação](#observação)

## Instalação

Para rodar o código, você precisa ter o Python instalado junto com as bibliotecas necessárias. Você pode instalar as bibliotecas com o seguinte comando:

```bash
pip install tensorflow numpy scikit-learn flask matplotlib seaborn pillow
```

Certifique-se de que seus dados de imagem estão organizados da seguinte forma:

```css
data/
    train/
        dermatite/
        normal/
    validation/
        dermatite/
        normal/
    test/
        dermatite/
        normal/
```

## Descrição do Código

O código é dividido em várias partes que lidam com diferentes aspectos do projeto, desde a definição do modelo até o treinamento e a criação de uma API para predições.

## Treinamento do Modelo

### Importação de Bibliotecas

Vamos começar importando as bibliotecas necessárias para nosso projeto:

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16
from sklearn.utils import class_weight
import os
```

- **TensorFlow (tf)**: Principal biblioteca para criar e treinar redes neurais.
- **NumPy (np)**: Biblioteca para manipulação eficiente de arrays e cálculos numéricos.
- **Keras (parte do TensorFlow)**: Interface de alto nível para construir e treinar redes neurais.
- **ImageDataGenerator**: Utilizado para carregar e pré-processar imagens, aplicando transformações como rotação e mudança de brilho.
- **EarlyStopping e ModelCheckpoint**: Callbacks para interromper o treinamento cedo se o modelo não estiver melhorando e para salvar o melhor modelo durante o treinamento.
- **VGG16**: Um modelo de rede neural convolucional pré-treinado, que serve como base para nosso modelo.
- **Class_weight**: Função para calcular pesos das classes para lidar com desbalanceamento de dados.
- **OS**: Biblioteca para interagir com o sistema operacional, como ler arquivos.

### Definição da Semente Aleatória

Para garantir que nossos experimentos sejam reproduzíveis, definimos uma semente aleatória:

```python
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
```

- **set_seeds**: Função que define a semente aleatória para as bibliotecas NumPy e TensorFlow. Isso garante que os resultados sejam os mesmos toda vez que o código for executado.

### Criação do Modelo CNN

Vamos criar o modelo de rede neural convolucional (CNN):

```python
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
```

### Treinamento e Avaliação do Modelo

Agora vamos definir a função para treinar e avaliar o modelo:

```python
def train_and_evaluate_model():
    set_seeds()

    # Aumento de Dados
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
        class_mode='binary'
    )

    # Pesos das Classes para lidar com Desbalanceamento
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    best_val_accuracy = 0
    best_model = None

    while True:
        print("\nTraining model iteration")

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

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = cnn_model
            best_model.save('bovino_dermatite_model_final.keras')

            test_loss, test_accuracy = best_model.evaluate(test_generator)
            print(f"Perda no teste: {test_loss}")
            print(f"Acurácia no teste: {test_accuracy}")

        if val_accuracy >= 0.9:  # Se a acurácia de validação atingir 90%, interrompa o treinamento
            break

if __name__ == '__main__':
    train_and_evaluate_model()
```

## API com Flask

Para realizar predições em novas imagens, criamos uma API Flask.

### Importação de Bibliotecas

```python
import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
```

### Criação da API Flask

```python
app = Flask(__name__)
model = None
```

### Função de Predição de Imagens

```python
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = int(prediction[0][0] > 0.5)
    true_class = 1 if 'dermatite' in img_path.lower() else 0
    accuracy = np.round(prediction[0][0], 2) if predicted_class == true_class else np.round(1 - prediction[0][0], 2)

    result = "dermatite" if predicted_class == 1 else "normal"
    return result, predicted_class, accuracy
```

### Endpoint de Predição

```python
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file provided"}), 400

    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    result, predicted_class, accuracy = predict_image(filepath)
    os.remove(filepath)

    return jsonify({
        "predicted_class": predicted_class,
        "result": result,
        "accuracy": accuracy
    }), 200
```

### Inicialização do Servidor

```python
if __name__ == '__main__':
    model_path = '

bovino_dermatite_model_final.keras'
    if os.path.exists(model_path):
        model = load_model(model_path)
    app.run(host='0.0.0.0', port=5000)
```

## Uso

Para usar a API, envie uma requisição POST para o endpoint `/predict` com um arquivo de imagem.

```bash
curl -X POST -F 'file=@path_to_your_image.jpg' http://localhost:5000/predict
```

## Resultados

Os resultados do modelo podem ser visualizados durante o treinamento, onde são exibidas a perda e acurácia de validação e teste.

## Observação

Certifique-se de que o servidor Flask está em execução para realizar predições e de que o modelo `bovino_dermatite_model_final.keras` está salvo no diretório correto.

---

Este projeto é uma aplicação prática de aprendizado profundo para classificação de imagens médicas, adaptado para identificar dermatite nodular contagiosa em bovinos, utilizando uma abordagem baseada em redes neurais convolucionais e perceptrons multicamadas.