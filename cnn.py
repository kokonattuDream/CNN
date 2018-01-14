from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifer = Sequential()

#convolutional layer
classifer.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))

#Max Pooling
classifer.add(MaxPooling2D(pool_size=(2,2)))

#Flatteing
classifer.add(Flatten())

#Full Connection
classifer.add(Dense(activation="relu", units=128))
classifer.add(Dense(activation="sigmoid", units=128))

#Compile
classifer.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])