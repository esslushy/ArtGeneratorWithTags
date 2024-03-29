import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse
import json

from Model import buildDiscriminator, buildGenerator

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

# Set global step
globalStep = 0

# Standardize randomness
tf.random.set_seed(7)
np.random.seed(7)

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

# Get tags and load into a tensorflow dataset
tagDataset = tf.data.Dataset.from_tensor_slices(tf.cast(np.load(settings['labels']), tf.float32))#float 32 used for compatible typing
# Load names of all images in order with tags and then read and preprocess them
imageDataset = tf.data.Dataset.from_tensor_slices(np.load(settings['imageNames']))#Gets all image paths
imageDataset = imageDataset.map(loadAndPreprocessImage, num_parallel_calls=AUTOTUNE)# Places images into datast
dataset = tf.data.Dataset.zip((imageDataset, tagDataset))
# Prep dataset for training
dataset = dataset.cache()#This helps improve performance if data doesnt fit in memory
dataset = dataset.batch(settings['batchSize'])
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

def calculateMultiscaleStructuralSimilarity(labels, images1):
    """
    Calculates the similarity between pairs of images made by the generator. Returns a set of values in the range [0, 1] where the closer to
    1 means the more similar the images. Large values returned from this means there has been mode collapse in the generator. This will be used
    as an extra metric during training to make sure the generator is learning properly.
    """
    # Create a set of noise of size (batchSize, 100)
    noise = tf.random.normal((labels.shape[0], 100))
    # Generate 2nd set of images
    images2 = generator((noise, labels))
    # Calculate the Multiscale Structural Similarity. max_val is 2 because the images range is [-1, 1]
    return tf.image.ssim_multiscale(images1, images2, 2)

# Loss functions
def generatorLoss(fakeLogits):
    # Ones like because the label for real images is 1, and the generator wants to make its images as realistic as possible
    return tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(fakeLogits), fakeLogits)# Should be a shape of (batchSize, 1)

def discriminatorLoss(realLogits, fakeLogits):
    # Ones like because the label for real images is 1, and the discriminator wants to approach that with its predictions on the real images
    discriminatorRealLoss = tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(realLogits), realLogits)# Should be a shape of (batchSize, 1).
    # Zeros like because the label for fake images is 0, and the discriminator wants to approach that with its predictions on the generators images
    discriminatorFakeLoss = tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(fakeLogits), fakeLogits)# Should be a shape of (batchSize, 1).
    return discriminatorRealLoss, discriminatorFakeLoss

def discriminatorLabelLoss(realLabelLogits, fakeLabelLogits, labels):
    # According to paper both are compared to the same labels https://arxiv.org/pdf/1610.09585.pdf
    # Finds discriminator's ability to guess labels
    discriminatorRealLabelLoss = tf.nn.sigmoid_cross_entropy_with_logits(labels, realLabelLogits)
    # Finds both generator's ability to create the right labels and the discriminator's ability to guess them. 
    discriminatorFakeLabelLoss = tf.nn.sigmoid_cross_entropy_with_logits(labels, fakeLabelLogits)
    return discriminatorRealLabelLoss, discriminatorFakeLabelLoss

# Train step
@tf.function
def trainStep(images, labels, globalStep, writer):
    # Makes a random noise distribution of (batchSize, 100)
    noise = tf.random.normal((images.shape[0], 100))
    with tf.GradientTape() as generatorTape, tf.GradientTape() as discriminatorTape:
        # Build fake images
        fakeImages = generator((noise, labels))
        # Get discriminator predictions
        realPredictions, realLabelPredictions, realLogits, realLabelLogits = discriminator(images)
        fakePredictions, fakeLabelPredictions, fakeLogits, fakeLabelLogits = discriminator(fakeImages)
        # Calculate Multiscale Structural Similarity in Generator.
        ssim = calculateMultiscaleStructuralSimilarity(labels, fakeImages)
        # Calculate losses
        genLoss = generatorLoss(fakeLogits)
        discRealLoss, discFakeLoss = discriminatorLoss(realLogits, fakeLogits)
        discRealLabelLoss, discFakeLabelLoss = discriminatorLabelLoss(realLabelLogits, fakeLabelLogits, labels)
        # Sum Losses. 
        genTotalLoss = genLoss + discFakeLabelLoss
        discTotalLoss = discRealLoss + discFakeLoss + discRealLabelLoss + discFakeLabelLoss

    # Collect Gradients
    generatorGradients = generatorTape.gradient(genTotalLoss, generator.trainable_variables)
    discriminatorGradients = discriminatorTape.gradient(discTotalLoss, discriminator.trainable_variables)

    # Run Optimizers
    generatorOptimizer.apply_gradients(zip(generatorGradients, generator.trainable_variables))
    discriminatorOptimizer.apply_gradients(zip(discriminatorGradients, discriminator.trainable_variables))

    # Accumalate Metrics
    discriminatorRealImagesAccuracy.update_state(tf.ones_like(realPredictions), realPredictions)
    discriminatorFakeImagesAccuracy.update_state(tf.zeros_like(fakePredictions), fakePredictions)

    # Log to tensorboard
    with tf.device('/cpu:0'), writer.as_default():
        tf.summary.scalar('Discriminator_Real_Images_Loss', tf.reduce_mean(discRealLoss), step=globalStep)
        tf.summary.scalar('Discriminator_Fake_Images_Loss', tf.reduce_mean(discFakeLoss), step=globalStep)
        tf.summary.scalar('Discriminator_Real_Image_Labels_Loss', tf.reduce_mean(discRealLabelLoss), step=globalStep)
        tf.summary.scalar('Discriminator_And_Generator_Fake_Image_Labels_Loss', tf.reduce_mean(discFakeLabelLoss), step=globalStep)# Applies to both
        tf.summary.scalar('Discriminator_Total_Loss', tf.reduce_mean(discTotalLoss), step=globalStep)
        tf.summary.scalar('Discriminator_Real_Images_Accuracy', discriminatorRealImagesAccuracy.result(), step=globalStep)
        tf.summary.scalar('Discriminator_Fake_Images_Accuracy', discriminatorFakeImagesAccuracy.result(), step=globalStep)
        tf.summary.scalar('Generator_Realism_Loss', tf.reduce_mean(genLoss), step=globalStep)
        tf.summary.scalar('Generator_Total_Loss', tf.reduce_mean(genTotalLoss), step=globalStep)
        tf.summary.scalar('Generator_Mode_Collapse_Percentage', tf.reduce_mean(ssim), step=globalStep)
        tf.summary.image('Generated_Images', fakeImages, max_outputs=8, step=globalStep)

# Summary Writer
writer = tf.summary.create_file_writer(settings['tensorboardLocation'])
# Checkpoint Model
checkpoint = tf.train.Checkpoint(generatorOptimizer=generatorOptimizer, discriminatorOptimizer=discriminatorOptimizer,
                                generator=generator, discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, directory=settings['saveModel'] + 'checkpoint_', max_to_keep=3, checkpoint_name='ckpt_epoch')#Keep only last 3 checkpoints of model

# Training
for epoch in range(settings['epochs']):
    print('On Epoch: ', epoch)
    for images, labels in dataset:
        # Train model and update tensorboard
        trainStep(images, labels, globalStep, writer)
        # Increment global step
        globalStep+=1

        with tf.device('/cpu:0'):
            # Checkpoint model each epoch
            manager.save(checkpoint_number=epoch)
            # Reset metrics so that they accumalate per epoch instead of over the entire training period
            discriminatorRealImagesAccuracy.reset_states()
            discriminatorFakeImagesAccuracy.reset_states()

# Save Final trained models in keras model format for easy reuse
tf.saved_model.save(generator, settings['saveModel'] + 'generator')
tf.saved_model.save(discriminator, settings['saveModel'] + 'discriminator')