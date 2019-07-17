import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
import argparse
import json

from Model import *

parser = argparse.ArgumentParser(description='Train the model for detecting false positives')
parser.add_argument('--epochs', type=int, help='The number of epochs to train for', default=150)
parser.add_argument('--batch_size', type=int, help='The batch size to train on', default=32)
parser.add_argument('--learning_rate', type=float, help='The learning rate of the model', default=0.002)
parser.add_argument('--images', type=str, help='The location to the folder containing the images', default='./dataset/images/')
parser.add_argument('--tags', type=str, help='The location to the .npy containing the labels', default='./dataset/tags.npy')
parser.add_argument('--tensorboard', type=str, help='The location to save the .info file for Tensorboard', default='./info')
parser.add_argument('--save_model', type=str, help='The location to save the model files to during training and at the end', default='./model')
parser.add_argument('--settings', type=str, help='Path to the json files with the settings. Use this instead of passing arguments to make it easier to rerun tests with the same values.', required=True)
arguments = parser.parse_args()

# Gather settings from json
with open(arguments.settings, 'r') as f:
    settings = json.load(f)

# Set global variables
gloablStep = 0

# Standardize randomness
tf.random.set_seed(7)
np.random.seed(7)

# Set prefetch buffer for dataloading
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Build dataset
def loadAndPreprocessImage(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, dtype=tf.float32)
    image = tf.image.resize(image, (256, 256))
    # Normalize to [-1, 1] range
    image = image - (255.0/2.0)
    image = image / (255.0/2.0)
    return image

# Get tags and load into a tensorflow dataset
tagDataset = tf.data.Dataset.from_tensor_slices(tf.cast(np.load(settings['tags']), tf.float32))#float 32 used for compatible typing
# Get all locations of pictures and load into the dataset
imageRoot = pathlib.Path(settings['images'])#lets use get images from the folder
imageDataset = tf.data.Dataset.from_tensor_slices([str(path) for path in imageRoot.iterdir()])#Gets all image paths
imageDataset = imageDataset.map(loadAndPreprocessImage, num_parallel_calls=AUTOTUNE)# Places images into datast
dataset = tf.data.Dataset.zip((imageDataset, tagDataset))
# Prep dataset for training
dataset = dataset.cache(filename='./cache.tf-data')#This helps improve performance if data doesnt fit in memory
dataset = dataset.shuffle(5000000)
dataset = dataset.batch(settings['batchSize'])
dataset = dataset.prefetch(buffer_size=AUTOTUNE)

# Build models
with tf.device('/GPU:0'): # Gpu used for training
    generator = buildGenerator(training=True)
    discriminator = buildDiscriminator(training=True)

# Loss Object
bceLoss = keras.losses.BinaryCrossentropy()

#Optimizers
generatorOptimizer = keras.optimizers.Adam(settings['learningRate'])
discriminatorOptimizer = keras.optimizers.Adam(settings['learningRate'])

# Metrics
discriminatorRealImagesAccuracy = keras.metrics.BinaryAccuracy()
discriminatorFakeImagesAccuracy = keras.metrics.BinaryAccuracy()

# Loss functions
@tf.function
def generatorLoss(fakePredictions):
    # Ones like because the label for real images is 1, and the generator wants to make its images as realistic as possible
    return bceLoss(fakePredictions, tf.ones_like(fakePredictions))# Should be a shape of (batchSize, 1)

@tf.function
def discriminatorLoss(realPredictions, fakePredictions):
    # Ones like because the label for real images is 1, and the discriminator wants to approach that with its predictions on the real images
    discriminatorRealLoss = bceLoss(realPredictions, tf.ones_like(realPredictions))# Should be a shape of (batchSize, 1).
    # Zeros like because the label for fake images is 0, and the discriminator wants to approach that with its predictions on the generators images
    discriminatorFakeLoss = bceLoss(fakePredictions, tf.zeros_like(fakePredictions))# Should be a shape of (batchSize, 1).
    # Add losses together to pass to optimizer
    discriminatorTotalLoss = discriminatorRealLoss + discriminatorFakeLoss
    return discriminatorRealLoss, discriminatorFakeLoss, discriminatorTotalLoss

# Train step
@tf.function
def trainStep(images, labels):
    noise = tf.random.normal((images.shape[0], 100))#Makes a random noise distribution of (batchSize, 100)
    with tf.GradientTape() as tape:
        # Build fake images
        fakeImages = generator(noise, labels)
        # Get discriminator predictions
        realPredictions = discriminator(images, labels)
        fakePredictions = discriminator(fakeImages, labels)
        # Calculate losses
        genLoss = generatorLoss(fakePredictions)
        discRealLoss, discFakeLoss, discTotalLoss = discriminatorLoss(realPredictions, fakePredictions)

    # Collect Gradients
    generatorGradients = tape.gradient(genLoss, generator.trainable_variables)
    discriminatorGradients = tape.gradient(discTotalLoss, discriminator.trainable_variables)

    # Run Optimizers
    generatorOptimizer(zip(generatorGradients, generator.trainable_variables))
    discriminatorOptimizer(zip(discriminatorGradients, discriminator.trainable_variables))

    # Accumalate Metrics
    discriminatorRealImagesAccuracy.update_state(tf.ones_like(realPredictions), realPredictions)
    discriminatorFakeImagesAccuracy.update_state(tf.zeros_like(fakePredictions), fakePredictions)

    # Log to tensorboard
    tf.summary.scalar('Discriminator_Real_Images_Loss', tf.reduce_mean(discRealLoss), step=gloablStep)
    tf.summary.scalar('Discriminator_Fake_Images_Loss', tf.reduce_mean(discFakeLoss), step=gloablStep)
    tf.summary.scalar('Discriminator_Real_Images_Accuracy', discriminatorRealImagesAccuracy.result().numpy(), step=gloablStep)
    tf.summary.scalar('Discriminator_Fake_Images_Accuracy', discriminatorFakeImagesAccuracy.result().numpy(), step=gloablStep)
    tf.summary.scalar('Generator_Loss', tf.reduce_mean(genLoss), step=gloablStep)
    tf.summary.image('Generated_Images', fakeImages, max_outputs=8, step=gloablStep)

# Checkpoint Model
checkpoint = tf.train.Checkpoint(generatorOptimizer=generatorOptimizer, discriminatorOptimizer=discriminatorOptimizer,
                                generator=generator, discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, directory=settings['saveModel'] + 'checkpoints', max_to_keep=3, checkpoint_name='ckpt_epoch')#Keep only last 3 checkpoints of model

# Summary Writer
writer = tf.summary.create_file_writer(settings['tensorboardLocation'])

# Training
with writer.as_default(): # All summaries made during training will be saved to this writer
    for epoch in range(settings['epochs']):
        for images, labels in dataset:
            # Train model and update tensorboard
            trainStep(images, labels)
            # Increment global step for logging to tensorboard
            gloablStep+=1

        # Checkpoint model each epoch
        manager.save(checkpoint_number=epoch)
        # Reset metrics so that they accumalate per epoch instead of over the entire training period
        discriminatorRealImagesAccuracy.reset_states()
        discriminatorFakeImagesAccuracy.reset_states()
        
# Save Final trained models in keras model format for easy reuse
tf.saved_model.save(generator, settings['saveModel'] + 'generator')
tf.saved_model.save(discriminator, settings['saveModel'] + 'discriminator')