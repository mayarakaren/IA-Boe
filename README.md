# Bovino Dermatite Classificação

Este repositório contém um projeto de aprendizado profundo usando TensorFlow e Keras para classificar imagens de bovinos, determinando se apresentam ou não dermatite nodular contagiosa. O modelo usa uma Rede Neural Convolucional (CNN) para realizar essa classificação.

## Índice

- [Bovino Dermatite Classificação](#bovino-dermatite-classificação)
  - [Índice](#índice)
  - [Instalação](#instalação)
  - [Descrição do Código](#descrição-do-código)
    - [Importação de Bibliotecas](#importação-de-bibliotecas)
    - [Definição do Modelo CNN](#definição-do-modelo-cnn)
    - [Compilação do Modelo](#compilação-do-modelo)
    - [Aumento de Dados](#aumento-de-dados)
    - [Geradores de Dados](#geradores-de-dados)
    - [Pesos de Classes](#pesos-de-classes)
    - [Callbacks](#callbacks)
    - [Treinamento e Avaliação do Modelo](#treinamento-e-avaliação-do-modelo)
    - [Função de Predição](#função-de-predição)
  - [Uso](#uso)
  - [Resultados](#resultados)
  - [Observação](#observação)

## Instalação

Para rodar o código, você precisa ter o Python instalado junto com as bibliotecas necessárias. Você pode instalar as bibliotecas com o seguinte comando:

```bash
pip install tensorflow numpy scikit-learn flask matplotlib seaborn pillow
```

Certifique-se de que seus dados de imagem estão organizados da seguinte forma:

```
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

### Importação de Bibliotecas

A primeira parte do código importa as bibliotecas necessárias para construir e treinar a CNN.

```python
import os
import numpy as np
import base64
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
```

- `os`: Para interações com o sistema operacional.
- `numpy`: Para operações numéricas.
- `base64`: Para manipulação de dados codificados em base64.
- `Flask`: Para construção da API.
- `tensorflow.keras`: Para construção e carregamento do modelo de aprendizado profundo.
- `PIL`: Para manipulação de imagens.

### Definição do Modelo CNN

A função `create_model` define a arquitetura da CNN usando o modelo pré-treinado VGG16.

```python
def create_model():
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout

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

    return model
```

- `VGG16`: Modelo pré-treinado utilizado como base.
- `GlobalAveragePooling2D`, `Dense`, `BatchNormalization`, `Dropout`: Camadas adicionais para refinar a saída do modelo.

### Compilação do Modelo

O modelo é compilado com o otimizador Adam e a função de perda `binary_crossentropy`, adequada para classificação binária. A métrica de avaliação usada é a acurácia.

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Aumento de Dados

O aumento de dados é realizado para melhorar a generalização do modelo, gerando variações das imagens de treino.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
```

### Geradores de Dados

Os geradores de dados são configurados para carregar as imagens de treino, validação e teste.

```python
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    'data/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)
```

### Pesos de Classes

Os pesos das classes são calculados para lidar com o desbalanceamento dos dados, atribuindo maior peso às classes menos representadas.

```python
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}
```

### Callbacks

Os callbacks são usados para salvar o melhor modelo e interromper o treinamento se o desempenho não melhorar após várias épocas.

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

checkpoint_cb = ModelCheckpoint('bovino_dermatite_model_best.keras', save_best_only=True, monitor='val_accuracy', mode='max')
early_stopping_cb = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```

### Treinamento e Avaliação do Modelo

O treinamento e a avaliação do modelo são realizados até que a acurácia de validação atinja um ponto de referência satisfatório.

```python
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    class_weight=class_weights,
    callbacks=[checkpoint_cb, early_stopping_cb]
)

model.save('bovino_dermatite_model_final.keras')

model = load_model('bovino_dermatite_model_best.keras')
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy:.4f}')
```

### Função de Predição

A função `predict_image` é usada para realizar predições em novas imagens.

```python
def predict_image(model, img_array):
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    accuracy = float(prediction[0][0]) if prediction[0][0] > 0.5 else float(1 - prediction[0][0])
    predicted_class = 'Dermatite' if prediction[0][0] > 0.5 else 'Normal'
    
    return predicted_class, accuracy
```

- `predict_image`: Realiza a predição e retorna a classe com maior probabilidade e a acurácia.

## Uso

Para treinar o modelo, use:

```bash
python train_model.py
```

Para iniciar o servidor Flask e realizar predições em imagens novas, use:

```bash
python app.py
```

A API espera uma requisição POST no endpoint `/predict` com um JSON contendo a imagem codificada em base64 e, opcionalmente, o nome do arquivo para inferência da classe verdadeira.

Exemplo de uso da API:

```json
{
    "image": "<base64_string>",
    "filename": "dermatite_example.jpg"
}
```

## Resultados

Os resultados do modelo, incluindo a matriz de confusão e o relatório de classificação, são exibidos após o treinamento e a avaliação. A matriz de confusão e o relatório de classificação fornecem uma visão detalhada do desempenho do modelo em termos de precisão, recall, f1-score e acurácia para cada classe.

## Observação

Certifique-se de que suas imagens estão organizadas corretamente nas diretorias de treino, validação e teste. Ajuste os hiperparâmetros conforme necessário para obter o melhor desempenho possível para seu conjunto de dados específico.