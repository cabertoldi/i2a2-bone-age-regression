import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint   
from keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalMaxPooling2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

def init_model(patientSex, df):
    if not os.path.isfile(f'./data/model.hand.x-ray.weights.{patientSex}.best.hdf5'):
        images, outputs = prepare_dataset(df)
        model = _train(patientSex, images, outputs)
        print('New train!')
    else:
        model = _create_model()
        model.load_weights(f'./data/model.hand.x-ray.weights.{patientSex}.best.hdf5')
        print('Using network trained!')

    return model

def prepare_dataset(df):
    images = [_preprocess_images(f"./data/clean-images/{filename}") for filename in df['fileName']]
    images = np.array(images, dtype=np.float32)

    outputs = df['boneage']
    outputs = np.array(outputs, dtype=np.float32)

    return images, outputs

def _preprocess_images(filename):
    image = load_img(filename, target_size=(128, 128))
    image = img_to_array(image)
    image = image.reshape((image.shape[0], image.shape[1], image.shape[2]))
    return preprocess_input(image) 

def _create_model():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape= (128, 128, 3))
    
    number_of_frozen_layers = 0
    for i, layer in enumerate(base_model.layers):
      if i>=number_of_frozen_layers:
        break
      layer.trainable = False

    x = GlobalMaxPooling2D()(base_model.output)
    x = Flatten()(x)
    x = Dense(16, activation = 'relu')(x)
    x = Dense(1, activation = 'linear')(x)

    model = Model(base_model.input, x)
    model.summary()
    
    return model

def _train(patientSex, images, outputs):
    print('Number images:', len(images))
    print('Number outputs:', len(outputs))

    # divindo dataset de treinamento em treinamento, teste e validação
    seed = 42
    x_train, x_test, y_train, y_test = train_test_split(images, outputs, test_size = 0.20, random_state=seed)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.20, random_state = seed)

    # normalização
    x_train = x_train.astype('float32')/255
    x_valid = x_valid.astype('float32')/255
    x_test = x_test.astype('float32')/255

    # mudando escala de idades para valores entre [0-1]
    max_bornage = outputs.max()
    y_train = y_train / max_bornage
    y_valid = y_valid / max_bornage
    y_test = y_test / max_bornage

    model = _create_model()
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    checkpointer = [ModelCheckpoint(filepath=f'./data/model.hand.x-ray.weights.{patientSex}.best.hdf5', save_best_only=True),
                    EarlyStopping(patience= 5)]
    history = model.fit(x_train, y_train,
           batch_size=32,
           epochs=50,
           verbose=1,
           validation_data=(x_valid, y_valid),
           callbacks=checkpointer)
    
    plt.plot(history.history['mse'])
    plt.show()
    
    # carregando os pesos que geraram a melhor precisão de validação
    model.load_weights(f'./data/model.hand.x-ray.weights.{patientSex}.best.hdf5')

    # avaliar e imprimir a precisão do teste
    loss, mse = model.evaluate(x_test, y_test, verbose=2)
    print("Testing set Mean Square Error: {:5.2f} MPG".format(mse))

    return model