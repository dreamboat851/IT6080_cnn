import tensorflow
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.image import imread



physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

data_dir='C:\\Users\\sajee\\OneDrive\\Documents\\myMSc\\AI_IT5100\\Assignment2\\cxrmod'

test_path=data_dir+"\\test\\"
train_path=data_dir+"\\training\\"

image_shape=(299,299,3)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_gen=ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.1,
                             zoom_range=0.1,
                             horizontal_flip=True,
                             fill_mode='nearest',
                             rescale=1/255)

print(image_gen.flow_from_directory(test_path))
print(image_gen.flow_from_directory(train_path))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten

model=Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

early_stop=EarlyStopping(monitor='val_loss',patience=4)

batch_size=16

train_image_gen=image_gen.flow_from_directory(train_path,
                                              target_size=image_shape[:2],
                                              color_mode='rgb',
                                              batch_size=batch_size,
                                              class_mode='categorical')

test_image_gen=image_gen.flow_from_directory(test_path,
                                              target_size=image_shape[:2],
                                              color_mode='rgb',
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              shuffle=False)

results=model.fit_generator(train_image_gen,
                            epochs=12,
                            validation_data=test_image_gen,
                            callbacks=[early_stop])

from pathlib import Path

# Save neural network structure
model_structure = model.to_json()
f = Path("COVID_CRX1_structure.json")
f.write_text(model_structure)

# Save neural network's trained weights
model.save_weights("COVID_CRX1_weights.h5")

# Save whole model to one file
model.save("f")

import pickle
file_name='history1.txt'
f = open(file_name,'wb')
pickle.dump(results.history,f)
f.close()

accuracy_val = results.history['val_accuracy']
loss_val = results.history['val_loss']
epochs = range(1,13)
plt.plot(epochs, accuracy_val, 'g', label='Validation accuracy')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Validation accuracy and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print(train_image_gen.class_indices)

from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

json_file = Path('COVID_CRX1_structure.json')

model_structure = json_file.read_text()

model_1 = model_from_json(model_structure)

model_1.load_weights("COVID_CRX1_weights.h5")

from tensorflow.keras.preprocessing import image

my_image = image.load_img('Normal-1042.png', target_size=image_shape)

my_img_arr = image.img_to_array(my_image)

my_image_arr = my_img_arr/255

my_img_arr.shape

my_img_arr = np.expand_dims(my_img_arr, axis=0)

my_img_arr.shape

model_1.predict(my_img_arr)

