from tensorflow import keras

# Generator
class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.reshape = keras.layers.Reshape((-1, 4, 4, 1024), input_shape=(-1, 100))#Converts the random noise into the primary shape to be operated on 

    def call(self, noise, labels):


# Discriminator