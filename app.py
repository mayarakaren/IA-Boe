import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight

def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

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
