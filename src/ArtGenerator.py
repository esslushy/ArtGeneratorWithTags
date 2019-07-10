import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
import argparse
import json

parser = argparse.ArgumentParser(description='Train the model for detecting false positives')
parser.add_argument('--epochs', type=int, help='The number of epochs to train for', default=150)
parser.add_argument('--batch_size', type=int, help='The batch size to train on', default=32)
parser.add_argument('--learning_rate', type=float, help='The learning rate of the model', default=0.002)
parser.add_argument('--images', type=str, help='The location to the folder containing the images', default='./dataset/images/')
parser.add_argument('--tags', type=str, help='The location to the .npy containing the labels', default='./dataset/tags.npy')
parser.add_argument('--tensorboard', type=str, help='The location to save the .info file for Tensorboard', default='./info')
parser.add_argument('--save_model', type=str, help='The location to save the model files to during training and at the end', default='./model')
parser.add_argument('--settings', type=str, help='Path to the json files with the settings', required=True)
arguments = parser.parse_args()

#Gather settings from json
with open(arguments.settings, 'r') as f:
    settings = json.load(f)

#Set global variables
settings['noiseLength'] = 100
gloablStep = 0

#Standardize randomness
tf.random.set_seed(7)
np.random.seed(7)

#Set prefetch buffer for dataloading
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

#Get tags and load into a tensorflow dataset
tagDataset = tf.data.Dataset.from_tensor_slices(tf.cast(np.load(settings['tags']), tf.float32))#float 32 used for compatible typing
#Get all locations of pictures and load into the dataset
imageRoot = pathlib.Path(settings['images'])#lets use get images from the folder
imageDataset = tf.data.Dataset.from_tensor_slices([str(path) for path in imageRoot.iterdir()])#Gets all image paths
imageDataset = imageDataset.map(loadAndPreprocessImage, num_parallel_calls=AUTOTUNE)# Places images into datast
dataset = tf.data.Dataset.zip((imageDataset, tagDataset))
# Prep dataset for training
dataset = dataset.cache(filename='./cache.tf-data')#This helps improve performance if data doesnt fit in memory
dataset = dataset.shuffle(5000000)
dataset = dataset.batch(settings['batchSize'])
dataset = dataset.prefetch(buffer_size=AUTOTUNE)