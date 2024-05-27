# Bovino Dermatite Classificação - README Detalhado

Este repositório contém um projeto de aprendizado profundo usando TensorFlow e Keras para classificar imagens de bovinos, determinando se apresentam ou não dermatite nodular contagiosa. O modelo usa uma Rede Neural Convolucional (CNN) para realizar essa classificação.

## Índice

- [Bovino Dermatite Classificação - README Detalhado](#bovino-dermatite-classificação---readme-detalhado)
  - [Índice](#índice)
  - [Instalação](#instalação)
  - [Descrição do Código](#descrição-do-código)
    - [Importação de Bibliotecas](#importação-de-bibliotecas)
    - [Definição da CNN](#definição-da-cnn)
    - [Compilação do Modelo](#compilação-do-modelo)
    - [Aumento de Dados](#aumento-de-dados)
    - [Geradores de Dados](#geradores-de-dados)
    - [Callbacks](#callbacks)
    - [Treinamento do Modelo](#treinamento-do-modelo)
    - [Avaliação do Modelo](#avaliação-do-modelo)
    - [Salvando o Modelo](#salvando-o-modelo)
    - [Função de Predição](#função-de-predição)
  - [Uso](#uso)
  - [Resultados](#resultados)
  - [Observação](#observação)

## Instalação

Para rodar o código, você precisa ter o Python instalado junto com as bibliotecas necessárias. Você pode instalar as bibliotecas com o seguinte comando:

```bash
pip install tensorflow numpy
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

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
```

- `tensorflow` e `tensorflow.keras`: Biblioteca de aprendizado profundo usada para construir e treinar a CNN.
- `Sequential`: Permite a criação de um modelo sequencial.
- `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Dropout`, `BatchNormalization`: Camadas usadas para construir a arquitetura da CNN.
- `ImageDataGenerator`: Ferramenta para gerar dados de imagem com aumento (augmentation).
- `load_img`, `img_to_array`: Funções para carregar e processar imagens.
- `numpy`: Biblioteca para operações numéricas.

### Definição da CNN

```python
cnn_model = Sequential()

cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Dropout(0.25))

cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Dropout(0.25))

cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Dropout(0.25))

cnn_model.add(Flatten())

cnn_model.add(Dense(256, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(1, activation='sigmoid'))
```

### Compilação do Modelo

```python
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
```

### Aumento de Dados

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)
```

### Geradores de Dados

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
    class_mode='binary'
)
```

### Callbacks

```python
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('bovino_dermatite_model_best.keras', save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
```

### Treinamento do Modelo

```python
history = cnn_model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[checkpoint_cb, early_stopping_cb]
)
```

### Avaliação do Modelo

```python
val_loss, val_accuracy = cnn_model.evaluate(validation_generator)
print(f"Perda na validação: {val_loss}")
print(f"Acurácia na validação: {val_accuracy}")

test_loss, test_accuracy = cnn_model.evaluate(test_generator)
print(f"Perda no teste: {test_loss}")
print(f"Acurácia no teste: {test_accuracy}")
```

### Salvando o Modelo

```python
cnn_model.save('bovino_dermatite_model_final.keras')
```

### Função de Predição

```python
def predict_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = cnn_model.predict(img_array)
    return 'Dermatite' if prediction[0][0] > 0.5 else 'Normal'

result = predict_image('data/train/normal/Normal_Skin_24.png')
print(f"Resultado da predição: {result}")
```

## Uso

1. **Preparação dos Dados**: Certifique-se de que as imagens estão organizadas nas pastas de treino, validação e teste.
2. **Treinamento**: Execute o código para treinar o modelo. O modelo treinado será salvo como `bovino_dermatite_model_final.keras`.
3. **Predição**: Use a função `predict_image` para classificar novas imagens.

## Resultados

Após o treinamento, o modelo será avaliado em termos de perda e acurácia tanto no conjunto de validação quanto no conjunto de teste. Exemplos de uso da função de predição também são fornecidos para testar novas imagens.

Este README fornece uma visão detalhada do código, explicando cada parte e seu propósito. Use este guia para entender melhor como o modelo funciona e como usá-lo para classificar imagens de bovinos com dermatite nodular contagiosa.

## Observação

O script é configurado para treinar e avaliar o modelo continuamente em um loop infinito. Certifique-se de que isso é intencional, pois pode consumir muitos recursos de computação. Ajuste conforme necessário para seu ambiente de desenvolvimento ou produção.

```python
while True:
    train_and_evaluate_model()
```

