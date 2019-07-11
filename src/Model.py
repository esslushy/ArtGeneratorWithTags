import tensorflow as tf
from tensorflow import keras

# Custom Layers
def ConvolutionalConcat(inputs, addition):
    with tf.name_scope('Convolutional_Concat'):
        addition = keras.layers.RepeatVector(inputs.shape[1] * inputs.shape[2])(addition)# Repeat addition so that it can be reshaped into a 3d shape and appended
        addition = keras.layers.Reshape((inputs.shape[1], inputs.shape[2], addition.shape[-1]))(addition)# Reshapes so it can be added. addition.shape[-1] is the channel width
        return keras.layers.Concatenate()((inputs, additions))# Add the resized additions to the channels or last axis

def TransposeConvolutionLayer(x, filters, kernelSize, strides):
    x = keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernelSize, strides=strides, padding='same')(x)
    x = keras.layers.BatchNormalization()(x, training=True)# Training is on this time
    return keras.layers.LeakyReLU()(x)# Leaky relu sometimes performs better than relu, but relu is the norm
    
def ConvolutionLayer(x, filters, kernelSize, strides):
    x = keras.layers.Conv2D(filters=filters, kernel_size=kernelSize, strides=strides, padding='same')(x)
    x = keras.layers.BatchNormalization()(x, training=True)# Training is on this time
    return keras.layers.LeakyReLU()(x)# Leaky relu sometimes performs better than relu, but relu is the norm

# Generator Builder
def buildGenerator():
    with tf.name_scope('Generator'):
        with tf.name_scope('Reshape_and_Project_Inputs'):
            noiseInputs = keras.Input(shape=(None, 100))#Noise input
            tagInputs = keras.Input(shape=(None, 20))#Tag input
            noiseAndTags = keras.layers.Concatenate()([noiseInputs, tagInputs])# Adds on tag depth
            noise = keras.layers.Dense(4*4*1024)(noiseAndTags)# Project it across enough values to give enough attributes for a 3d shape
            reshapedNoise = keras.layers.Reshape((4, 4, 1024))(noise)# Reshape noise into a 3d shape
        with tf.name_scope('Convolutional_Transpose_Layer_1'):
            x = ConvolutionalConcat(reshapedNoise, tagInputs) # Append tags back to data
            x = TransposeConvolutionLayer(x, 512, 5, 2)#Upsample 4x4x1024 -> 8x8x512
        with tf.name_scope('Convolutional_Transpose_Layer_2'):
            x = ConvolutionalConcat(x, tagInputs) # Append tags back to data
            x = TransposeConvolutionLayer(x, 256, 5, 2)#Upsample 8x8x512 -> 16x16x256
        with tf.name_scope('Convolutional_Transpose_Layer_3'):
            x = ConvolutionalConcat(x, tagInputs) # Append tags back to data
            x = TransposeConvolutionLayer(x, 128, 5, 2)#Upsample 16x16x256 -> 32x32x128
        with tf.name_scope('Convolutional_Transpose_Layer_4'):
            x = ConvolutionalConcat(x, tagInputs) # Append tags back to data
            x = TransposeConvolutionLayer(x, 64, 5, 2)#Upsample 32x32x128 -> 64x64x64
        with tf.name_scope('Convolutional_Transpose_Layer_5'):
            x = ConvolutionalConcat(x, tagInputs) # Append tags back to data
            x = TransposeConvolutionLayer(x, 32, 5, 2)#Upsample 64x64x64 -> 128x128x32
        with tf.name_scope('Convolutional_Transpose_Layer_6'):
            x = ConvolutionalConcat(x, tagInputs) # Append tags back to data
            x = TransposeConvolutionLayer(x, 16, 5, 2)#Upsample 128x128x32 -> 256x256x16
        with tf.name_scope('Convolutional_Transpose_Layer_7'):
            x = ConvolutionalConcat(x, tagInputs) # Append tags back to data
            x = TransposeConvolutionLayer(x, 8, 5, 1)#Upsample 256x256x16 -> 256x256x8
        with tf.name_scope('Convolutional_Transpose_Layer_8'):
            x = ConvolutionalConcat(x, tagInputs) # Append tags back to data
            x = TransposeConvolutionLayer(x, 4, 5, 1)#Upsample 256x256x8 -> 256x256x4
        with tf.name_scope('Convolutional_Transpose_Layer_9'):
            x = ConvolutionalConcat(x, tagInputs) # Append tags back to data
            x = keras.layers.Conv2DTranspose(filters=3, kernel_size=5, strides=1, padding='same')(x)# Upsample 256x256x4 -> 256x256x3
            # No Batch Norm or Leaky Relu in last layer
        with tf.name_scope('Output'):
            x = keras.activations.tanh(x) # Reduces to [-1, 1] range
    generator = keras.Model(inputs=[noiseInputs, tagInputs]) # Construct model
    return generator
    
# Discriminator Builder
def buildDiscriminator():
    