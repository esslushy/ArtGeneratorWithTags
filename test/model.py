import tensorflow as tf
from tensorflow import keras

def ResizeConvolutionLayer(x, size, filters):
    x = tf.image.resize(x, size)
    x = keras.layers.Conv2D(filters=filters, kernel_size=4, strides=1, padding='same', 
                            kernel_initializer=tf.keras.initializers.GlorotNormal(7))(x)
    return x

def ResizeConvolutionLayerWithNorm(x, size, filters, training):
    x = ResizeConvolutionLayer(x, size, filters)
    x = keras.layers.BatchNormalization()(x, training=training)# Training is on this time
    return keras.layers.LeakyReLU(alpha=0.2)(x)# Relu for generator

def buildGenerator(training):
    noiseInputs = keras.Input(shape=(100))# Noise input
    noise = keras.layers.Dense(4*4*1024)(noiseInputs)# Project it across enough values to give enough attributes for a 3d shape
    reshapedNoise = keras.layers.Reshape((4, 4, 1024))(noise)# Reshape noise into a 3d shape
    x = ResizeConvolutionLayerWithNorm(reshapedNoise, (8, 8), 512, training)# 4x4x1024 -> 8x8x512
    x = ResizeConvolutionLayerWithNorm(x, (16, 16), 256, training)# 8x8x512 -> 16x16x256
    x = ResizeConvolutionLayerWithNorm(x, (32, 32), 128, training)# 16x16x256 -> 32x32x128
    x = ResizeConvolutionLayerWithNorm(x, (64, 64), 64, training)# 32x32x128 -> 64x64x64
    x = ResizeConvolutionLayerWithNorm(x, (128, 128), 32, training)# 64x64x64 -> 128x128x32
    x = ResizeConvolutionLayer(x, (256, 256), 3)# 128x128x32 -> 256x256x3
    # No batch norm in last layer
    # Use tanh to normalize between -1 and 1 instead of Leaky ReLU
    outImages = keras.layers.Activation(keras.activations.tanh)(x)
    generator = keras.Model(inputs=[noiseInputs], outputs=[outImages])
    return generator

def buildDiscriminator(training):
    imageInputs = keras.Input(shape=(256, 256, 3))# Image Inputs
    x = ResizeConvolutionLayer(imageInputs, (128, 128), 32)# 256x256x3 -> 128x128x32
    # No batch norm in first layer according to DCGAN paper
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = ResizeConvolutionLayerWithNorm(x, (64, 64), 64, training)# 128x128x32 -> 64x64x64
    x = ResizeConvolutionLayerWithNorm(x, (32, 32), 128, training)# 64x64x64 -> 32x32x128
    x = ResizeConvolutionLayerWithNorm(x, (16, 16), 256, training)# 32x32x128 -> 16x16x256
    x = ResizeConvolutionLayerWithNorm(x, (8, 8), 512, training)# 16x16x256 -> 8x8x512
    x = ResizeConvolutionLayerWithNorm(x, (4, 4), 1024, training)# 8x8x512 -> 4x41024
    #Output uses sigmoid for classification no leaky relu
    x = keras.layers.Flatten()(x) # Flatten out inputs into a 1d array of (Batch Size, 4x4x1024)
    logits = keras.layers.Dense(1)(x)
    out = keras.layers.Activation(keras.activations.sigmoid)(logits) # Compress into 1 output label between 0 (fake image) and 1 (real image) for classification
    discriminator = keras.Model(inputs=[imageInputs], outputs=[out, logits])# Build model
    return discriminator
