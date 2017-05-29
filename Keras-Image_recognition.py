import keras
import numpy as np
from parser import *
from tensorflow.examples.tutorials.mnist import input_data

# step 1 - Collect data
 training_data = load_data("C:\data\images\training")
 validation_data = load_data("C:\data\images\validation")

# step 2 -Build model
model = Sequential() # alternative is Graph
model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])

# Step 3 - Train model
model.fit_generator(training_data, 
samples_per_epoch=2048, 
nb_epoch=30,
validation_data=validation_data,
nb_val_samples=832)

model.save_weights("C:\data\images\simple_CNN.h5")

# Test model
img = image.load_img("img001.jpg", target_size = (224, 224))
prediction = model.predict(img)
print(prediction)
