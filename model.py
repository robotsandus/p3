import csv
import cv2
import numpy as np
import random
#import pandas as pd
#from pandas import DataFrame
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 


lines = []
with open('udacity_data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(lines)
    while 1: # Loop forever so the generator never terminates
        shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                counter = 0
                if counter == 0:
                    counter = counter + 1
                    continue
                name = 'udacity_data/data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

ch, row, col = 3, 160, 320  # Trimmed image format



batch_size = 128

images = []
steering_angle_measurements = []

counter = 0
for line in lines:
    if counter == 0:
        counter = counter + 1
        continue
    
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'udacity_data/data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    steering_angle_measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(steering_angle_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, ELU, Activation, ZeroPadding2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam


model = Sequential()
#model.add(ZeroPadding2D((1, 1), input_shape=(ch,row, col)))
#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
#model.add(Lambda(lambda x: (x/127.5 - 1.)))
model.add((Convolution2D(32, 3,3 ,border_mode='same', subsample=(2,2), name='conv1')))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1), name='pool1'))
#model.add(Convolution2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D())

#model.add(Flatten())
#model.add(Dense(120))
#model.add(Dense(84))
#model.add(Dense(1))

#model.compile(loss='mse',optimizer='adam')



model.add(Convolution2D(64, 3,3 ,border_mode='same',subsample=(2,2), name='conv2'))
model.add(Activation('relu',name='relu2'))
model.add(MaxPooling2D(pool_size=(2,2), name='pool2'))

model.add(Convolution2D(128, 3,3,border_mode='same',subsample=(1,1), name='conv3'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2,2), name='pool3'))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(128, name='dense1'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128, name='dense2'))

model.add(Dense(1,name='output'))

model.compile(optimizer=Adam(lr= 0.0001), loss="mse")


history_object = model.fit(X_train,y_train, validation_split=0.3,shuffle=True,nb_epoch=5)
#history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples),validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)

#history_object = model.fit_generator()


#history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples),validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model_udacity_1.h5')
