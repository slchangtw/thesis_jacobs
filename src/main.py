import numpy as np
import pandas as pd

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, LeakyReLU
from keras.optimizers import Adamax
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

from datetime import datetime

import cv2

def rgb_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
import json

if __name__ == '__main__':
    IMAGE_HEIGHT = 64
    IMAGE_WIDTH = 64
    IMAGE_CHANNELS = 3
    BATCH_SIZE = 512
    EPOCHS = 10
    lr = 0.002

    train_dir = 'data/input/Training/'
    test_dir = 'data/input/Test/'
    
    train_datagen = ImageDataGenerator(rescale=1./255 
                                       validation_split=0.05)

    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        subset='validation')
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=1,
        shuffle=False)
    
    inv_class_map = {v: k for k, v in train_generator.class_indices.items()}
    num_classes = len(inv_class_map)
    
    # define the model
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))
    

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))
    
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    
    # callback list
    st = datetime.now().strftime('%H%m%d%m%y')

    callbacks_list = [
            ModelCheckpoint(
                filepath='model/' + st + '.h5',
                monitor='val_acc',
                save_best_only=True,
            ),
            CSVLogger(
                filename='log/' + st + '.log'
            )
            #LearningRateScheduler(lambda epoch, lr: lr / (1 + epoch))
    ]
    
    # initiate the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adamax(lr=lr),
                  metrics=['accuracy'])
    
    # start training
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples//BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples//BATCH_SIZE,
    )
    
    model = load_model('model/' + st + '.h5')
    
    score = model.evaluate_generator(test_generator, steps=test_generator.samples, verbose=1)
    print('Test accuracy:', score[1])
    
    params = {
        'model': 'model/' + st + '.h5',
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': lr,
        'activation': 'relu',
        'BN': True,
        'dropout': False,
        'relu_first': True,
        'hsv_tranformation': True,
        'lr_scheduler': False,
        'test_acc': score[1]
    }
    
    with open('params/' + st + '.json', 'w') as fp:
        json.dump(params, fp)
    
