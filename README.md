# Bovino Dermatite Classificação

Este repositório contém um projeto de aprendizado profundo usando TensorFlow e Keras para classificar imagens de bovinos, determinando se apresentam ou não dermatite nodular contagiosa. O modelo usa uma Rede Neural Convolucional (CNN) para realizar essa classificação.

## Índice

- [Bovino Dermatite Classificação](#bovino-dermatite-classificação)
  - [Índice](#índice)
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
pip install tensorflow numpy scikit-learn
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
```

- `tensorflow` e `tensorflow.keras`: Bibliotecas de aprendizado profundo usadas para construir e treinar a CNN.
- `Sequential`: Permite a criação de um modelo sequencial.
- `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Dropout`, `BatchNormalization`: Camadas usadas para construir a arquitetura da CNN.
- `ImageDataGenerator`: Ferramenta para gerar dados de imagem com aumento (augmentation).
- `EarlyStopping`, `ModelCheckpoint`: Callbacks para controle do treinamento.
- `class_weight`: Função para computar pesos de classes desbalanceadas.
- `numpy`: Biblioteca para operações numéricas.

### Definição da Função para Configuração de Sementes

A função `set_seeds` define a semente para tornar os resultados reprodutíveis, garantindo que os experimentos possam ser repetidos com os mesmos resultados.

```python
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
```

### Definição do Modelo CNN

A função `create_model` define a arquitetura da CNN. Este modelo é composto por várias camadas de convolução, normalização em lotes, pooling, dropout, e camadas densas totalmente conectadas.

```python
def create_model():
    model = Sequential([
        # Camadas CNN
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        
        # Camadas MLP
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

- `Conv2D`: Camada de convolução para extrair características das imagens.
- `BatchNormalization`: Normaliza a saída da camada anterior para acelerar o treinamento e melhorar a estabilidade do modelo.
- `MaxPooling2D`: Reduz a dimensionalidade espacial (downsampling), diminuindo a quantidade de parâmetros e computação na rede.
- `Dropout`: Regularização para prevenir overfitting, desconectando aleatoriamente unidades da rede durante o treinamento.
- `Flatten`: Achata a entrada para uma dimensão, preparando-a para a camada densa.
- `Dense`: Camada totalmente conectada para a classificação final.

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
    class_mode='binary'
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

O treinamento e a avaliação do modelo são realizados em um loop que continua até que uma condição de parada seja atendida.

```python
def train_and_evaluate_model():
    set_seeds()

    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=30,
        width

_shift_range=0.2,
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

    # Class weights to handle imbalance
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    best_val_accuracy = 0
    best_model = None

    iteration = 0
    while True:
        iteration += 1
        print(f"\nTraining model iteration {iteration}")

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
        print(f"Iteration {iteration} - Perda na validação: {val_loss}")
        print(f"Iteration {iteration} - Acurácia na validação: {val_accuracy}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = cnn_model

            best_model.save('bovino_dermatite_model_final.keras')

            test_loss, test_accuracy = best_model.evaluate(test_generator)
            print(f"Perda no teste: {test_loss}")
            print(f"Acurácia no teste: {test_accuracy}")

        # Adicionando condição para predição a cada 5 iterações
        if iteration % 5 == 0:
            def predict_image(img_path):
                img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0

                prediction = best_model.predict(img_array)
                return 'Dermatite' if prediction[0][0] > 0.5 else 'Normal'

            result = predict_image('data/train/dermatite/Lumpy_Skin_316.png')
            print(f"Resultado da predição: {result}")

        # Interrompe se atingir 10 iterações sem melhorias
        if iteration >= 10:
            break

train_and_evaluate_model()
```

- `set_seeds`: Configura a semente para reprodutibilidade.
- `train_generator`, `validation_generator`, `test_generator`: Geradores de dados para treino, validação e teste.
- `class_weights`: Calcula os pesos das classes para lidar com desbalanceamento.
- `checkpoint_cb`, `early_stopping_cb`: Callbacks para salvar o melhor modelo e interromper o treinamento antecipadamente se necessário.
- `cnn_model.fit`: Treina o modelo por até 15 épocas.
- `cnn_model.evaluate`: Avalia o modelo nos dados de validação e teste.
- `predict_image`: Função interna para predizer a classe de uma imagem específica a cada 5 iterações.
- O loop de treinamento é interrompido após 10 iterações sem melhorias significativas.

### Função de Predição

A função `predict_image` permite que o modelo faça predições em novas imagens.

```python
def predict_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = best_model.predict(img_array)
    return 'Dermatite' if prediction[0][0] > 0.5 else 'Normal'

result = predict_image('data/train/dermatite/Lumpy_Skin_316.png')
print(f"Resultado da predição: {result}")
```

- `load_img`: Carrega uma imagem do caminho especificado.
- `img_to_array`: Converte a imagem carregada para um array NumPy.
- `np.expand_dims`: Adiciona uma dimensão extra para representar o lote.
- A imagem é normalizada para o intervalo [0, 1].
- `predict`: Faz a predição usando o modelo treinado.
- A função retorna 'Dermatite' se a previsão for maior que 0.5, caso contrário, retorna 'Normal'.

## Uso

1. **Preparação dos Dados**: Certifique-se de que as imagens estão organizadas nas pastas de treino, validação e teste.
2. **Treinamento**: Execute o código para treinar o modelo. O modelo treinado será salvo como `bovino_dermatite_model_final.keras`.
3. **Predição**: Use a função `predict_image` para classificar novas imagens.

## Resultados

Após o treinamento, o modelo será avaliado em termos de perda e acurácia tanto no conjunto de validação quanto no conjunto de teste. Exemplos de uso da função de predição também são fornecidos para testar novas imagens.

## Observação

O script é configurado para treinar e avaliar o modelo continuamente em um loop infinito. Certifique-se de que isso é intencional, pois pode consumir muitos recursos de computação. Ajuste conforme necessário para seu ambiente de desenvolvimento ou produção.