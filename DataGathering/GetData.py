import pandas as pd
import numpy as np
import requests
import shutil

# Number to increment to download images
imageNum = 0

#Tags
tags = []

# Image downloader function
def downloadImage(image):
    global imageNum
    try:
        response = requests.get(image, stream=True)
    except:
        print('No response from website. This will just catch it and continue downloading')
        return False
    #Only save if got response
    if(response.status_code == 200):
        with open('../dataset/images/{}.jpg'.format(imageNum), 'wb') as f: #Precents writing an empty file if no data is there to populate it
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, f)
            print('Got image {} from {}'.format(imageNum, image))
            imageNum += 1
            return True
    else:
        return False


# Download images and store tags
def getImagesAndTags(row):
    success = downloadImage(row['src'])
    if(success): # only add tag if we got image
        tags.append(row.drop('src').values)#gets all values but source, so all the tags

#Read data csv into pandas
linksAndData = pd.read_csv('../dataset/info.csv')
print(linksAndData.columns)
# Run function to collect data into Tensorflow readable format
linksAndData.apply(getImagesAndTags, axis=1)
#save tags
np.save(tags, '../dataset/tags.npy')