import tensorflow as tf
from model import buildGenerator
import matplotlib.pyplot as plt

generator = buildGenerator(True)
noise = tf.random.normal((1, 100))
with tf.device('/cpu:0'):
    image = generator(noise).numpy()[0]

image = image /2
image = image + 0.5

plt.imshow(image)
plt.show()