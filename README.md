# ArtGeneratorWithTags
A remake of my old Abstract Art Generator in Tensorflow 2.0 that accepts user inputs when generating images.

# Requirements
- A dedicated gpu from nvidia.
- python 3.6.7+
- Tensorflow Gpu 2.0+
- pathlib
An easy install requirements.txt is included so that you can run pip -r install requirements.txt.

# Setup
Gather a dataset with the type of images that you desire and their appropiate labels. Make sure to download all your images to a location in the same order as the tags appear in their numpy file. So 1.jpg would correspond to the first index in tags.npy and 2.jpg would respond to the second and so on so that they are read by pathlib in numerical order and zipped in the right numerical order with their tags. An example can be found in my GetData.py where I download images and collect tags together into a folder and array at the same time and save them both so they are in order.
Next all you need to do is navigate to settings.json and change them how you like. If you need more information run python ArtGenerator.py --help to learn more about each of the settings. Once you have them setup run python Artgenerator.py --settings settings.json and it will begin training. This should take a while, but depending on your setup it may go faster or slower.

# Note
The folder `basic_gan` is a dcgan made to build off of when building one using tags. This also includes an experiment with another type of convolution layer called a resize convolution which is suppose to give better results without the necessary 1x1 conv layers which will help increase batch size and reduce training time.