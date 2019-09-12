import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse
import json

from model import buildDiscriminator, buildGenerator

parser = argparse.ArgumentParser(description='Train the model for detecting false positives')
parser.add_argument('--epochs', type=int, help='The number of epochs to train for', default=150)
parser.add_argument('--batchSize', type=int, help='The batch size to train on', default=32)
parser.add_argument('--learningRate', type=float, help='The learning rate of the model', default=0.002)
parser.add_argument('--images', type=str, help='The location to the folder containing the images', default='../dataset/images/')
parser.add_argument('--labels', type=str, help='The location to the .npy containing the labels', default='../dataset/labels.npy')
parser.add_argument('--tensorboardLocation', type=str, help='The location to save the .info file for Tensorboard', default='./info')
parser.add_argument('--saveModel', type=str, help='The location to save the model files to during training and at the end', default='./model')
parser.add_argument('--settings', type=str, help='Path to the json files with the settings. Use this instead of passing arguments to make it easier to rerun tests with the same values.', required=True)
arguments = parser.parse_args()

# Gather settings from json
with open(arguments.settings, 'r') as f:
    settings = json.load(f)

# Standardize randomness
tf.random.set_seed(7)
np.random.seed(7)

# Setup global step
globalStep = tf.Variable(initial_value=0, dtype=tf.int64)

# Setup Summary Writer
writer = tf.summary.create_file_writer(settings['tensorboardLocation'])

# Set prefetch buffer for dataloading
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Build dataset
def loadAndPreprocessImage(path):
    # Load image
    image = tf.io.read_file(settings['images'] + path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    # Normalize to [-1, 1] range
    image = image - (255.0/2.0)
    image = image / (255.0/2.0)
    return image

# Load names of all images in order with tags and then read and preprocess them
dataset = tf.data.Dataset.from_tensor_slices(np.load(settings['imageNames'], allow_pickle=True))#Gets all image paths
dataset = dataset.map(loadAndPreprocessImage, num_parallel_calls=AUTOTUNE)# Places images into datast
# Prep dataset for training
dataset = dataset.cache()#This helps improve performance
dataset = dataset.batch(settings['batchSize'], drop_remainder=True)
dataset = dataset.prefetch(buffer_size=AUTOTUNE)

# Build models
generator = buildGenerator(training=True)
discriminator = buildDiscriminator(training=True)

#Optimizers
generatorOptimizer = keras.optimizers.Adam(settings['learningRate'], beta_1=0.5)
discriminatorOptimizer = keras.optimizers.Adam(settings['learningRate'], beta_1=0.5)

# Metrics
discriminatorRealImagesAccuracy = keras.metrics.BinaryAccuracy()
discriminatorFakeImagesAccuracy = keras.metrics.BinaryAccuracy()

# Tensorboard logging
def logToTensorboard(discRealLoss, discFakeLoss, discTotalLoss, genLoss, genSimilarityLoss, genTotalLoss, ssim, fakeImages):
    with writer.as_default(): # Necessary for images. Helps reduce gpu load
        tf.summary.scalar('Discriminator_Real_Images_Loss', tf.reduce_mean(discRealLoss), step=globalStep)
        tf.summary.scalar('Discriminator_Fake_Images_Loss', tf.reduce_mean(discFakeLoss), step=globalStep)
        tf.summary.scalar('Discriminator_Total_Loss', tf.reduce_mean(discTotalLoss), step=globalStep)
        tf.summary.scalar('Discriminator_Real_Images_Accuracy', discriminatorRealImagesAccuracy.result(), step=globalStep)
        tf.summary.scalar('Discriminator_Fake_Images_Accuracy', discriminatorFakeImagesAccuracy.result(), step=globalStep)
        tf.summary.scalar('Generator_Realism_Loss', tf.reduce_mean(genLoss), step=globalStep)
        tf.summary.scalar('Generator_Mode_Collapse_Loss', tf.reduce_mean(genSimilarityLoss), step=globalStep)
        tf.summary.scalar('Generator_Total_Loss', tf.reduce_mean(genTotalLoss), step=globalStep)
        tf.summary.scalar('Generator_Mode_Collapse_Percentage', tf.reduce_mean(ssim), step=globalStep)
        tf.summary.image('Generated_Images', fakeImages, max_outputs=8, step=globalStep)

        # Increment step
        globalStep.assign_add(1)

def calculateMultiscaleStructuralSimilarity(images1):
    """
    Calculates the similarity between pairs of images made by the generator. Returns a set of values in the range [0, 1] where the closer to
    1 means the more similar the images. Large values returned from this means there has been mode collapse in the generator. This will be used
    as an extra metric during training to make sure the generator is learning properly.
    """
    # Create a set of noise of size (batchSize, 100)
    noise = tf.random.normal((settings['batchSize'], 100))
    # Generate 2nd set of images
    images2 = generator(noise)
    # Calculate the Multiscale Structural Similarity. max_val is 2 because the images range is [-1, 1]
    return tf.image.ssim_multiscale(images1, images2, 2, filter_size=8)

# Loss functions
def generatorLoss(fakeLogits, ssim):
    # Ones like because the label for real images is 1, and the generator wants to make its images as realistic as possible
    genLoss = tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(fakeLogits), fakeLogits)# Should be a shape of (batchSize, 1)
    ssimLoss = tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(ssim), ssim)
    return genLoss, ssimLoss

def discriminatorLoss(realLogits, fakeLogits):
    # Ones like because the label for real images is 1, and the discriminator wants to approach that with its predictions on the real images
    discriminatorRealLoss = tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(realLogits), realLogits)# Should be a shape of (batchSize, 1).
    # Zeros like because the label for fake images is 0, and the discriminator wants to approach that with its predictions on the generators images
    discriminatorFakeLoss = tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(fakeLogits), fakeLogits)# Should be a shape of (batchSize, 1).
    return discriminatorRealLoss, discriminatorFakeLoss

# Train step
@tf.function(input_signature=[tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32)])
def trainStep(images):
    # Makes a random noise distribution of (batchSize, 100)
    noise = tf.random.normal((settings['batchSize'], 100))
    with tf.GradientTape() as generatorTape, tf.GradientTape() as discriminatorTape:
        # Build fake images
        fakeImages = generator(noise)
        # Get discriminator predictions
        realPredictions, realLogits = discriminator(images)
        fakePredictions, fakeLogits = discriminator(fakeImages)
        # Calculate Multiscale Structural Similarity in Generator.
        ssim = calculateMultiscaleStructuralSimilarity(fakeImages)
        # Calculate losses
        genLoss, genSimilarityLoss = generatorLoss(fakeLogits, ssim)
        discRealLoss, discFakeLoss = discriminatorLoss(realLogits, fakeLogits)
        # Sum Losses. 
        genTotalLoss = genLoss + genSimilarityLoss
        discTotalLoss = discRealLoss + discFakeLoss

    # Collect Gradients
    generatorGradients = generatorTape.gradient(genTotalLoss, generator.trainable_variables)
    discriminatorGradients = discriminatorTape.gradient(discTotalLoss, discriminator.trainable_variables)

    # Run Optimizers
    generatorOptimizer.apply_gradients(zip(generatorGradients, generator.trainable_variables))
    discriminatorOptimizer.apply_gradients(zip(discriminatorGradients, discriminator.trainable_variables))

    # Accumalate Metrics
    discriminatorRealImagesAccuracy.update_state(tf.ones_like(realPredictions), realPredictions)
    discriminatorFakeImagesAccuracy.update_state(tf.zeros_like(fakePredictions), fakePredictions)
    # Return values to be logged
    return discRealLoss, discFakeLoss, discTotalLoss, genLoss, genSimilarityLoss, genTotalLoss, ssim, fakeImages

# Setup Training Checkpoints
checkpoint = tf.train.Checkpoint(generatorOptimizer=generatorOptimizer, discriminatorOptimizer=discriminatorOptimizer,
                                generator=generator, discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, directory=settings['saveModel'], max_to_keep=3, checkpoint_name='ckpt_epoch')#Keep only last 3 checkpoints of model

# Restore model if exists
if settings['restore']:
    # Run everything once to create variables
    trainStep(next(iter(dataset)))
    checkpoint.restore(manager.latest_checkpoint).assert_consumed()

# Training
for epoch in range(settings['epochs']):
    print(f'Epoch: {epoch}')
    with tf.device('/cpu:0'): # Ensures everything is run off gpu except for the actual training
        for batchNum, images in dataset.enumerate():
            # Train model
            with tf.device('/gpu:0'):
                discRealLoss, discFakeLoss, discTotalLoss, genLoss, genSimilarityLoss, genTotalLoss, ssim, fakeImages = trainStep(images)

            if batchNum % 150 == 0:
                logToTensorboard(discRealLoss, discFakeLoss, discTotalLoss, genLoss, genSimilarityLoss, genTotalLoss, ssim, fakeImages)

            # Checkpoint model each epoch
            tf.py_function(manager.save, [], [tf.string])
            # Reset metrics so that they accumalate per epoch instead of over the entire training period
            discriminatorRealImagesAccuracy.reset_states()
            discriminatorFakeImagesAccuracy.reset_states()

# Save Final trained models in saved model format for easy reuse
tf.saved_model.save(generator, settings['saveModel'] + 'generator')
tf.saved_model.save(discriminator, settings['saveModel'] + 'discriminator')