# Bovino Dermatite Classificação

Este repositório contém um projeto de aprendizado profundo usando TensorFlow e Keras para classificar imagens de bovinos, determinando se apresentam ou não dermatite nodular contagiosa. O modelo usa uma Rede Neural Convolucional (CNN) para realizar essa classificação.

## Índice

- [Instalação](#instalação)
- [Descrição do Código](#descrição-do-código)
  - [Importação de Bibliotecas](#importação-de-bibliotecas)
  - [Definição da Função para Configuração de Sementes](#definição-da-função-para-configuração-de-sementes)
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
pip install tensorflow numpy scikit-learn flask matplotlib seaborn
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
```

- `tensorflow` e `tensorflow.keras`: Bibliotecas de aprendizado profundo usadas para construir e treinar a CNN.
- `Sequential`, `load_model`: Permite a criação e carregamento de modelos sequenciais.
- `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Dropout`, `BatchNormalization`, `GlobalAveragePooling2D`: Camadas usadas para construir a arquitetura da CNN.
- `ImageDataGenerator`, `img_to_array`, `load_img`: Ferramentas para gerar e processar dados de imagem.
- `EarlyStopping`, `ModelCheckpoint`: Callbacks para controle do treinamento.
- `VGG16`: Modelo pré-treinado usado como base para a CNN.
- `class_weight`: Função para computar pesos de classes desbalanceadas.
- `confusion_matrix`, `classification_report`: Ferramentas para avaliação de modelo.
- `matplotlib.pyplot`, `seaborn`: Bibliotecas para visualização de dados.
- `Flask`: Framework para construção da API.
- `os`, `argparse`: Bibliotecas padrão para manipulação de arquivos e parsing de argumentos.

### Definição da Função para Configuração de Sementes

A função `set_seeds` define a semente para tornar os resultados reprodutíveis, garantindo que os experimentos possam ser repetidos com os mesmos resultados.

```python
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
```

### Definição do Modelo CNN

A função `create_model` define a arquitetura da CNN usando o modelo pré-treinado VGG16.

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

- `VGG16`: Modelo pré-treinado utilizado como base.
- `GlobalAveragePooling2D`: Reduz a dimensionalidade da saída da VGG16.
- `Dense`, `BatchNormalization`, `Dropout`: Camadas adicionais para refinar a saída do modelo.
- `sigmoid`: Função de ativação usada para a saída binária.

### Compilação do Modelo

O modelo é compilado com o otimizador Adam e a função de perda `binary_crossentropy`, adequada para classificação binária. A métrica de avaliação usada é a acurácia.

```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

### Aumento de Dados

O aumento de dados é realizado para melhorar a generalização do modelo, gerando variações das imagens de treino.

```python
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

- `rescale`: Normaliza os valores dos pixels para o intervalo [0, 1].
- `shear_range`, `zoom_range`, `horizontal_flip`, `rotation_range`, `width_shift_range`, `height_shift_range`, `brightness_range`: Parâmetros para aumentar os dados, criando variações das imagens originais para aumentar a robustez do modelo.

### Geradores de Dados

Os geradores de dados são configurados para carregar as imagens de treino, validação e teste.

```python
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
    class_mode='binary',
    shuffle=False  # Importante para garantir que as predições e rótulos correspondam corretamente
)
```

- `flow_from_directory`: Cria geradores que carregam imagens das diretorias especificadas e aplicam as transformações definidas no `ImageDataGenerator`.
- `target_size`: Redimensiona todas as imagens para 224x224 pixels.
- `batch_size`: Número de imagens carregadas por vez durante o treinamento.
- `class_mode`: Define a classificação binária.

### Pesos de Classes

Os pesos das classes são calculados para lidar com o desbalanceamento dos dados, atribuindo maior peso às classes menos representadas.

```python
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}
```

- `compute_class_weight`: Calcula pesos para cada classe baseado na frequência das classes nos dados de treinamento.
- `class_weights`: Dicionário de pesos de classes usados durante o treinamento para balancear a influência de cada classe na função de perda.

### Callbacks

Os callbacks são usados para salvar o melhor modelo e interromper o treinamento se o desempenho não melhorar após várias épocas.

```python
checkpoint_cb = ModelCheckpoint('bovino_dermatite_model_best.keras', save_best_only=True, monitor='val_accuracy', mode='max')
early_stopping_cb = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```

- `ModelCheckpoint`: Salva o modelo sempre que a acurácia de validação melhora.
- `EarlyStopping`: Interrompe o treinamento se a perda de validação não melhorar após 5 épocas consecutivas, restaurando os pesos do melhor modelo.

### Treinamento e Avaliação do Modelo

O treinamento e a avaliação do modelo são realizados até que a acurácia de validação atinja um ponto de referência satisfatório.

```python
def train_and_evaluate_model():
    set_seeds()

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip

=True,
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
        class_mode='binary',
        shuffle=False  # Importante para garantir que as predições e rótulos correspondam corretamente
    )

    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    model = create_model()

    checkpoint_cb = ModelCheckpoint('bovino_dermatite_model_best.keras', save_best_only=True, monitor='val_accuracy', mode='max')
    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

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

    predictions = model.predict(test_generator)
    y_true = test_generator.classes
    y_pred = np.round(predictions).flatten()

    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=['normal', 'dermatite'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['normal', 'dermatite'], yticklabels=['normal', 'dermatite'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    print(cr)
```

### Função de Predição

A função `predict_image` é usada para realizar predições em novas imagens.

```python
def predict_image(model_path, image_path):
    model = load_model(model_path)
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    if prediction >= 0.5:
        return 'dermatite'
    else:
        return 'normal'
```

- `predict_image`: Carrega o modelo salvo e faz uma predição em uma imagem fornecida.
- `load_model`: Carrega o modelo salvo.
- `load_img`: Carrega a imagem e redimensiona para 224x224 pixels.
- `img_to_array`: Converte a imagem para um array.
- `expand_dims`: Adiciona uma dimensão extra para corresponder ao formato esperado pelo modelo.
- `predict`: Realiza a predição e retorna a classe com maior probabilidade.

## Uso

Para treinar o modelo, use:

```bash
python train_model.py
```

Para realizar predições em uma nova imagem, use:

```bash
python predict.py --model_path 'bovino_dermatite_model_best.keras' --image_path 'path_to_image.jpg'
```

## Resultados

Os resultados do modelo, incluindo a matriz de confusão e o relatório de classificação, são exibidos após o treinamento e a avaliação. A matriz de confusão e o relatório de classificação fornecem uma visão detalhada do desempenho do modelo em termos de precisão, recall, f1-score e acurácia para cada classe.

## Observação

Certifique-se de que suas imagens estão organizadas corretamente nas diretorias de treino, validação e teste. Ajuste os hiperparâmetros conforme necessário para obter o melhor desempenho possível para seu conjunto de dados específico.