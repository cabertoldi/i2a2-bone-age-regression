import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf

from keras.applications.vgg16 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint   
from keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

def init_model(patientSex):
    if not os.path.isfile(f'./data/model.hand.x-ray.weights.{patientSex}.best.hdf5'):
        model = _train(patientSex)
        print('New train!')
    else:
        model = _create_model()
        model.load_weights(f'./data/model.hand.x-ray.weights.{patientSex}.best.hdf5')
        print('Using network trained!')

    return model

def _preprocess_images(filename):
    image = load_img(filename, target_size=(256, 256))
    image = img_to_array(image)
    image = image.reshape((image.shape[0], image.shape[1], image.shape[2]))
    return preprocess_input(image)

def _prepare_dataset(patientSex):
    df = pd.read_csv('./data/train.csv')
    df_patientSex = df.query(f'patientSex == "{patientSex}"')

    images = [_preprocess_images(f"./data/clean-images-train/{filename}") for filename in df_patientSex['fileName']]
    images = np.array(images, dtype=np.float32)
    print(np.shape(images))

    outputs = df_patientSex['boneage']
    outputs = np.array(outputs, dtype=np.float32)

    return images, outputs

def _create_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu', input_shape=(256, 256, 3)))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="linear"))

    model.summary()

    return model

def _train(patientSex):
    images, outputs = _prepare_dataset(patientSex)
    print('Number images:', len(images))
    print('Number outputs:', len(outputs))

    # dividindo dados em dados de teste e treino
    seed = 42

    x_train, x_test, y_train, y_test = train_test_split(images, outputs, test_size = 0.2, random_state=seed)

    # normalização
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    # mudando escala de idades para valores entre [0-1]
    max_bornage = outputs.max()
    y_train = y_train / max_bornage
    y_test = y_test / max_bornage

    # divindo dataset de treinamento em treinamento, treino e validação
    (x_train, x_valid) = x_train[20:], x_train[:20]
    (y_train, y_valid) = y_train[20:], y_train[:20]

    model = _create_model()
    model.compile(loss="mean_absolute_percentage_error", optimizer=Adam(lr=1e-3, decay=1e-3 / 200), metrics=['accuracy'])

    checkpointer = [ModelCheckpoint(filepath=f'./data/model.hand.x-ray.weights.{patientSex}.best.hdf5', save_best_only=True),
                    EarlyStopping(patience=10)]
    hist = model.fit(x_train, y_train,
           epochs=100,
           validation_data=(x_valid, y_valid), 
           callbacks=checkpointer)

    # carregando os pesos que geraram a melhor precisão de validação
    model.load_weights(f'./data/model.hand.x-ray.weights.{patientSex}.best.hdf5')

    # avaliar e imprimir a precisão do teste
    score = model.evaluate(x_test, y_test, verbose=0)
    print('\n', 'Test accuracy:', score[1])

    return model
