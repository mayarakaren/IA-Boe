import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, image
import numpy as np

# Definindo a CNN
cnn_model = Sequential()

# Adicionando camadas de convolução e pooling
cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D((2, 2)))

# Flattening
cnn_model.add(Flatten())

# Adicionando as camadas do MLP
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dense(1, activation='sigmoid'))  # Usando sigmoid para classificação binária

# Compilando o modelo
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation para evitar overfitting
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Diretórios de treinamento, validação e teste
train_dir = 'data/train'
validation_dir = 'data/validation'
test_dir = 'data/test'

# Geradores de dados
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

# Treinando o modelo
history = cnn_model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator
)

# Avaliação no conjunto de validação
val_loss, val_accuracy = cnn_model.evaluate(validation_generator)
print(f"Perda na validação: {val_loss}")
print(f"Acurácia na validação: {val_accuracy}")

# Avaliação no conjunto de teste
test_loss, test_accuracy = cnn_model.evaluate(test_generator)
print(f"Perda no teste: {test_loss}")
print(f"Acurácia no teste: {test_accuracy}")

# Salvando o modelo treinado
cnn_model.save('bovino_dermatite_model.h5')

# Função para predição de novas imagens
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = cnn_model.predict(img_array)
    return 'Dermatite' if prediction[0][0] > 0.5 else 'Normal'

# Teste de uma nova imagem
result = predict_image('path/to/new_image.jpg')
print(f"Resultado da predição: {result}")
