import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generator(dir, gen=ImageDataGenerator(rescale=1./255), shuffle=True, batch_size=1, target_size=(24,24), class_mode='categorical'):
    return gen.flow_from_directory(
        dir,
        batch_size=batch_size,
        shuffle=shuffle,
        color_mode='grayscale',
        class_mode=class_mode,
        target_size=target_size
    )

BS = 32
TS = (24, 24)

training_set = generator('project/dataset_new/train', shuffle=True, batch_size=BS, target_size=TS)
test_set = generator('project/dataset_new/test', shuffle=True, batch_size=BS, target_size=TS)

SPE = len(training_set.classes) // BS
VS = len(test_set.classes) // BS

print(f"Steps per epoch: {SPE}, Validation steps: {VS}")

cnn = Sequential()

cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)))
cnn.add(MaxPooling2D(pool_size=(1,1)))  # Changed to (2,2) for actual downsampling

cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(1,1)))

cnn.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(1,1)))

cnn.add(Dropout(0.25))

cnn.add(Flatten())
cnn.add(Dense(128, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(4, activation='softmax'))

# Compile
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
cnn.fit(training_set, validation_data=test_set, epochs=15, steps_per_epoch=SPE, validation_steps=VS)

# Save the model
cnn.save('models/model.h5', overwrite=True)
