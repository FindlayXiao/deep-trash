import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D, Conv2D
from keras.applications import MobileNet
from keras.applications.densenet import DenseNet121
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers


def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

        ## As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


base_model = MobileNet(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(5))(x)
x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(5))(x)
preds = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)


for layer in model.layers[:-1]:
    layer.trainable = False

for layer in model.layers[-1:]:
    layer.trainable = True


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True, vertical_flip=True)
test_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)


train_generator = train_datagen.flow_from_directory('data/train', target_size=(224, 224),
                                                    color_mode='rgb', batch_size=50, class_mode='categorical',
                                                    shuffle=True)


validation_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(224, 224),
        batch_size=10,
        class_mode='categorical')



model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

step_size_train = train_generator.n//train_generator.batch_size

history = model.fit_generator(
        train_generator,
        steps_per_epoch=step_size_train,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=50)

plot_history(history)

model.save('deeptrash.h5')
