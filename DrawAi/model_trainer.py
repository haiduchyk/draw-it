import keras.layers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.utils import np_utils

import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pickle
from keras.callbacks import TensorBoard


class StopCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.999:
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True


def keras_model(image_x, image_y):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_of_classes, activation='softmax'))

    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Flatten(input_shape=(image_x, image_y, 1)),
    #     tf.keras.layers.Dense(512, activation=tf.nn.relu),
    #     tf.keras.layers.Dense(num_of_classes, activation=tf.nn.softmax)
    # ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])



    filepath = "models/QuickDraw.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return model, callbacks_list


def loadFromPickle():
    with open("divided_data/features", "rb") as f:
        features = np.array(pickle.load(f))
    with open("divided_data/labels", "rb") as f:
        labels = np.array(pickle.load(f))

    with open("divided_data/test_features", "rb") as f:
        test_features = np.array(pickle.load(f))
    with open("divided_data/test_labels", "rb") as f:
        test_labels = np.array(pickle.load(f))

    return features, test_features, labels, test_labels


def augmentData(features, labels):
    features = np.append(features, features[:, :, ::-1], axis=0)
    labels = np.append(labels, -labels, axis=0)
    return features, labels


def prepress_labels(labels):
    labels = np_utils.to_categorical(labels)
    return labels


def main():
    features, test_features, labels, test_labels = loadFromPickle()

    features, labels = shuffle(features, labels)
    test_features, test_labels = shuffle(test_features, test_labels)

    labels = prepress_labels(labels)
    test_labels = prepress_labels(test_labels)

    train_x = features.reshape(features.shape[0], 28, 28, 1)
    test_x = test_features.reshape(test_features.shape[0], 28, 28, 1)
    model, callbacks_list = keras_model(28, 28)
    model.summary()


    model.fit(train_x, labels, validation_data=(test_x, test_labels), epochs=30, batch_size=512,
              callbacks=[TensorBoard(log_dir="train_events"), StopCallback()])



    model.save('models/model.h5')


num_of_classes = 8
main()

# commands to analise training
# conda install -c conda-forge tensorboard
# tensorboard -- logdir=train_events
