import pandas as pd
import numpy as np
import requests

# Number to increment to download images
imageNum = 10000000

#Tags
tags = []

# Image downloader function
def downloadImage(image):
    with open('../dataset/images/{}.jpg'.format(imageNum), 'wb') as f:
        response = requests.get(image, stream=True)
        #Only save if got response
        if(response.ok):
            f.write(response.content)
            return True
        else:
            return False


# Download images and store tags
def getImagesAndTags(row):
    success = downloadImage(row['src'])
    if(success): # only add tag if we got image
        tags.append(row.drop('src', axis=1).values)#gets all values but source, so all the tags

#Read data csv into pandas
linksAndData = pd.read_csv('../dataset/info.csv')
# Run function to collect data into Tensorflow readable format
linksAndData.apply(getImagesAndTags)
#save tags
np.save(tags, '../dataset/tags.npy')