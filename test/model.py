import tensorflow as tf
from tensorflow import keras

def TransposeConvolutionLayer(x, filters, kernelSize, strides, training):
    x = keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernelSize, strides=strides, padding='same', 
                                    kernel_initializer=tf.keras.initializers.GlorotNormal(7))(x)
    x = keras.layers.BatchNormalization()(x, training=training)# Training is on this time
    return keras.layers.ReLU()(x)# Relu for generator
    
def ConvolutionLayer(x, filters, kernelSize, strides, training):
    x = keras.layers.Conv2D(filters=filters, kernel_size=kernelSize, strides=strides, padding='same', 
                            kernel_initializer=tf.keras.initializers.GlorotNormal(7))(x)
    x = keras.layers.BatchNormalization()(x, training=training)# Training is on this time
    return keras.layers.LeakyReLU(alpha=0.2)(x)# Leaky relu for discriminator

# Generator Builder
def buildGenerator(training):
    noiseInputs = keras.Input(shape=(100))# Noise input
    noise = keras.layers.Dense(4*4*1024)(noiseInputs)# Project it across enough values to give enough attributes for a 3d shape
    reshapedNoise = keras.layers.Reshape((4, 4, 1024))(noise)# Reshape noise into a 3d shape
    x = TransposeConvolutionLayer(reshapedNoise, 512, 4, 2, training)#Upsample 4x4x1024 -> 8x8x512
    x = TransposeConvolutionLayer(x, 256, 4, 2, training)#Upsample 8x8x512 -> 16x16x256
    x = TransposeConvolutionLayer(x, 128, 4, 2, training)#Upsample 16x16x256 -> 32x32x128
    x = TransposeConvolutionLayer(x, 64, 4, 2, training)#Upsample 32x32x128 -> 64x64x64
    x = TransposeConvolutionLayer(x, 32, 4, 2, training)#Upsample 64x64x64 -> 128x128x32
    x = TransposeConvolutionLayer(x, 16, 4, 1, training)#Upsample 128x128x32 -> 128x128x16
    x = TransposeConvolutionLayer(x, 8, 4, 1, training)#Upsample 128x128x16 -> 128x128x8
    x = TransposeConvolutionLayer(x, 4, 4, 1, training)#Upsample 128x128x8 -> 128x128x4
    x = keras.layers.Conv2DTranspose(filters=3, kernel_size=4, strides=1, padding='same')(x)# Upsample 128x128x4 -> 128x128x3
    # No Batch Norm or Leaky Relu in last layer
    outImages = keras.layers.Activation(keras.activations.tanh)(x) # Tanh used to normalize to [-1, 1] range
    generator = keras.Model(inputs=[noiseInputs], outputs=[outImages]) # Construct model
    return generator
    
# Discriminator Builder
def buildDiscriminator(training):
    imageInputs = keras.Input(shape=(128, 128, 3))#Noise input
    x = keras.layers.Conv2D(filters=4, kernel_size=5, strides=1, padding='same')(imageInputs)# 128x128x3 ->128x128x4
    # No batch norm in first layer according to the paper.
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = ConvolutionLayer(x, 8, 4, 1, training)# 128x128x4 -> 128x128x8
    x = ConvolutionLayer(x, 16, 4, 1, training)# 128x128x8 -> 128x128x16
    x = ConvolutionLayer(x, 32, 4, 1, training)# 128x128x16 -> 128x128x32
    x = ConvolutionLayer(x, 64, 4, 2, training)# 128x128x32 -> 64x64x64
    x = ConvolutionLayer(x, 128, 4, 2, training)# 64x64x64 -> 32x32x128
    x = ConvolutionLayer(x, 256, 4, 2, training)# 32x32x128 -> 16x16x256
    x = ConvolutionLayer(x, 512, 4, 2, training)# 16x16x256 -> 8x8x512
    x = keras.layers.Conv2D(filters=1024, kernel_size=4, strides=2, padding='same')(x)# 8x8x512 -> 4x4x1024
    x = keras.layers.BatchNormalization()(x, training=training)
    #Output uses sigmoid for classification no leaky relu
    x = keras.layers.Flatten()(x) # Flatten out inputs into a 1d array of (Batch Size, 4x4x1024)
    logits = keras.layers.Dense(1)(x)
    out = keras.layers.Activation(keras.activations.sigmoid)(logits) # Compress into 1 output label between 0 (fake image) and 1 (real image) for classification
    discriminator = keras.Model(inputs=[imageInputs], outputs=[out, logits])# Build model
    return discriminator
