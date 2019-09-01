import tensorflow as tf
from tensorflow import keras

def ResizeConvolutionLayer(x, size, filters):
    x = keras.layers.Lambda(tf.image.resize, arguments={'size' : size})(x)
    x = keras.layers.Conv2D(filters=filters, kernel_size=4, strides=1, padding='same', 
                            kernel_initializer=tf.keras.initializers.GlorotNormal(7))(x)
    return x

def ResizeConvolutionLayerWithNorm(x, size, filters, training):
    x = ResizeConvolutionLayer(x, size, filters)
    x = keras.layers.BatchNormalization()(x, training=training)# Training is on this time
    return keras.layers.ReLU()(x)# Relu for generator

def buildGenerator(training):
    noiseInputs = keras.Input(shape=(100))# Noise input
    noise = keras.layers.Dense(4*4*1024)(noiseInputs)# Project it across enough values to give enough attributes for a 3d shape
    reshapedNoise = keras.layers.Reshape((4, 4, 1024))(noise)# Reshape noise into a 3d shape
    x = ResizeConvolutionLayerWithNorm(reshapedNoise, (8, 8), 512, training)# 4x4x1024 -> 8x8x512
    x = ResizeConvolutionLayerWithNorm(reshapedNoise, (16, 16), 256, training)# 8x8x512 -> 16x16x256
    x = ResizeConvolutionLayerWithNorm(reshapedNoise, (32, 32), 128, training)# 16x16x256 -> 32x32x128
    x = ResizeConvolutionLayerWithNorm(reshapedNoise, (64, 64), 64, training)# 32x32x128 -> 64x64x64
    x = ResizeConvolutionLayerWithNorm(reshapedNoise, (128, 128), 32, training)# 64x64x64 -> 128x128x32
    x = ResizeConvolutionLayerWithNorm(reshapedNoise, (256, 256), 16, training)# 128x128x32 -> 256x256x16
    # No batch norm or resize in last layer
    x = keras.layers.Conv2D(filters=3, kernel_size=4, strides=1, padding='same', 
                            kernel_initializer=tf.keras.initializers.GlorotNormal(7))(x)
    # Use tanh to normalize between -1 and 1
    outImages = keras.layers.Activation(keras.activations.tanh)(x)
    model = keras.Model(inputs=[noiseInputs], outputs=[outImages])
    return model