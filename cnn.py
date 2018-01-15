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



#Fit to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('data/test_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

classifer.fit_generator(training_set,
                    steps_per_epoch=2000,
                    epochs=50,
                    validation_data=test_set,
                    validation_steps=800)