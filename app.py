import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np


def train_and_evaluate_model():
    # Definindo a CNN com uma arquitetura mais complexa
    cnn_model = Sequential()

    cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Dropout(0.25))  # Adicionando Dropout para evitar overfitting

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

    cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])

    # Data augmentation mais agressiva para evitar overfitting
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

    # Callbacks para salvar o melhor modelo e parar o treinamento cedo se não houver melhora
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('bovino_dermatite_model_best.keras', save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Treinando o modelo
    history = cnn_model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[checkpoint_cb, early_stopping_cb]
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
    cnn_model.save('bovino_dermatite_model_final.keras')

    # Função para predição de novas imagens
    def predict_image(img_path):
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = cnn_model.predict(img_array)
        return 'Dermatite' if prediction[0][0] > 0.5 else 'Normal'

    # Teste de uma nova imagem
    result = predict_image('data/train/normal/Normal_Skin_24.png')
    print(f"Resultado da predição: {result}")

# Loop infinito para executar o treinamento e avaliação continuamente
while True:
    train_and_evaluate_model()
