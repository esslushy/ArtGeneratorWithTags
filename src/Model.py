import tensorflow as tf
from tensorflow import keras

labelSize = 177

# Custom Layers
def ConvolutionalConcat(inputs, addition):
    with tf.name_scope('Convolutional_Concat'):
        addition = keras.layers.RepeatVector(inputs.shape[1] * inputs.shape[2])(addition)# Repeat addition so that it can be reshaped into a 3d shape and appended
        addition = keras.layers.Reshape((inputs.shape[1], inputs.shape[2], addition.shape[-1]))(addition)# Reshapes so it can be added. addition.shape[-1] is the channel width
        return keras.layers.Concatenate()([inputs, addition])# Add the resized additions to the channels or last axis

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
    with tf.name_scope('Generator'):
        with tf.name_scope('Reshape_and_Project_Inputs'):
            noiseInputs = keras.Input(shape=(100))# Noise input
            labelInputs = keras.Input(shape=(labelSize))# Label input.
            noiseAndLabels = keras.layers.Concatenate()([noiseInputs, labelInputs])# Adds on label depth
            noise = keras.layers.Dense(4*4*1024)(noiseAndLabels)# Project it across enough values to give enough attributes for a 3d shape
            reshapedNoise = keras.layers.Reshape((4, 4, 1024))(noise)# Reshape noise into a 3d shape
        with tf.name_scope('Convolutional_Transpose_Layer_1'):
            x = ConvolutionalConcat(reshapedNoise, labelInputs) # Append labels back to data
            x = TransposeConvolutionLayer(x, 512, 5, 2, training)#Upsample 4x4x1024 -> 8x8x512
        with tf.name_scope('Convolutional_Transpose_Layer_2'):
            x = ConvolutionalConcat(x, labelInputs) # Append labels back to data
            x = TransposeConvolutionLayer(x, 256, 5, 2, training)#Upsample 8x8x512 -> 16x16x256
        with tf.name_scope('Convolutional_Transpose_Layer_3'):
            x = ConvolutionalConcat(x, labelInputs) # Append labels back to data
            x = TransposeConvolutionLayer(x, 128, 5, 2, training)#Upsample 16x16x256 -> 32x32x128
        with tf.name_scope('Convolutional_Transpose_Layer_4'):
            x = ConvolutionalConcat(x, labelInputs) # Append labels back to data
            x = TransposeConvolutionLayer(x, 64, 5, 2, training)#Upsample 32x32x128 -> 64x64x64
        with tf.name_scope('Convolutional_Transpose_Layer_5'):
            x = ConvolutionalConcat(x, labelInputs) # Append labels back to data
            x = TransposeConvolutionLayer(x, 32, 5, 2, training)#Upsample 64x64x64 -> 128x128x32
        with tf.name_scope('Convolutional_Transpose_Layer_6'):
            x = ConvolutionalConcat(x, labelInputs) # Append labels back to data
            x = TransposeConvolutionLayer(x, 16, 5, 2, training)#Upsample 128x128x32 -> 256x256x16
        with tf.name_scope('Convolutional_Transpose_Layer_7'):
            x = ConvolutionalConcat(x, labelInputs) # Append labels back to data
            x = TransposeConvolutionLayer(x, 8, 5, 1, training)#Upsample 256x256x16 -> 256x256x8
        with tf.name_scope('Convolutional_Transpose_Layer_8'):
            x = ConvolutionalConcat(x, labelInputs) # Append labels back to data
            x = TransposeConvolutionLayer(x, 4, 5, 1, training)#Upsample 256x256x8 -> 256x256x4
        with tf.name_scope('Convolutional_Transpose_Layer_9'):
            x = ConvolutionalConcat(x, labelInputs) # Append labels back to data
            x = keras.layers.Conv2DTranspose(filters=3, kernel_size=5, strides=1, padding='same')(x)# Upsample 256x256x4 -> 256x256x3
            # No Batch Norm or Leaky Relu in last layer
        with tf.name_scope('Output'):
            outImages = keras.layers.Activation(keras.activations.tanh)(x) # Tanh used to normalize to [-1, 1] range
    generator = keras.Model(inputs=[noiseInputs, labelInputs], outputs=[outImages]) # Construct model
    return generator
    
# Discriminator Builder
def buildDiscriminator(training):
    with tf.name_scope('Discriminator'):
        with tf.name_scope('Input'):
            imageInputs = keras.Input(shape=(256, 256, 3))#Noise input
        with tf.name_scope('Convolutional_Layer_1'):
            x = keras.layers.Conv2D(filters=4, kernel_size=5, strides=1, padding='same')(imageInputs)# 256x256x3 ->256x256x4
            # No batch norm in first layer according to the paper.
            x = keras.layers.LeakyReLU(alpha=0.2)(x)
        with tf.name_scope('Convolutional_Layer_2'):
            x = ConvolutionLayer(x, 8, 5, 1, training)# 256x256x4 -> 256x256x8
        with tf.name_scope('Convolutional_Layer_3'):
            x = ConvolutionLayer(x, 16, 5, 1, training)# 256x256x8 -> 256x256x16
        with tf.name_scope('Convolutional_Layer_4'):
            x = ConvolutionLayer(x, 32, 5, 2, training)# 256x256x16 -> 128x128x32
        with tf.name_scope('Convolutional_Layer_5'):
            x = ConvolutionLayer(x, 64, 5, 2, training)# 128x128x32 -> 64x64x64
        with tf.name_scope('Convolutional_Layer_6'):
            x = ConvolutionLayer(x, 128, 5, 2, training)# 64x64x64 -> 32x32x128
        with tf.name_scope('Convolutional_Layer_7'):
            x = ConvolutionLayer(x, 256, 5, 2, training)# 32x32x128 -> 16x16x256
        with tf.name_scope('Convolutional_Layer_8'):
            x = ConvolutionLayer(x, 512, 5, 2, training)# 16x16x256 -> 8x8x512
        with tf.name_scope('Convolutional_Layer_9'):
            x = keras.layers.Conv2D(filters=1024, kernel_size=5, strides=2, padding='same')(x)# 8x8x512 -> 4x4x1024
            keras.layers.BatchNormalization()(x, training=training)
            #Output uses sigmoid for classification no leaky relu
        with tf.name_scope('Ouputs'):
            x = keras.layers.Flatten()(x) # Flatten out inputs into a 1d array of (Batch Size, 4x4x1024)
            logits = keras.layers.Dense(1)(x)
            labelsLogits = keras.layers.Dense(labelSize)(x)
            out = keras.layers.Dense(1, activation=keras.activations.sigmoid)(logits) # Compress into 1 output label between 0 (fake image) and 1 (real image) for classification
            labels = keras.layers.Dense(labelSize, activation=keras.activations.sigmoid)(labelsLogits)# Multiple classes are true so use sigmoid instead of softmax
    discriminator = keras.Model(inputs=[imageInputs], outputs=[out, labels, logits, labelsLogits])# Build model
    return discriminator
