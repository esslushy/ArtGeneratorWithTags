import sqlite3
import pandas as pd
import numpy as np
import urllib
import cv2

#connect to database with images and links
conn = sqlite3.connect('../dataset/BamImages.sqlite')
#read from tables with the tags and links
IDwithInfo = pd.read_sql(sql='select * from automatic_labels;', con=conn)
IDwithLink = pd.read_sql(sql='select * from modules;', con=conn)
print('read files')
#setup links and remove unnecessary colums
IDwithLink = IDwithLink.drop(['project_id', 'mature_content', 'license'], axis=1)
#merge them together on the mid which will line up links and tags
MergedInfo = pd.merge(IDwithInfo, IDwithLink, on='mid').drop('mid', axis=1)
#remove unused items hogging ram
del IDwithInfo
del IDwithLink
conn.close()
#check to make sure it worked
print(MergedInfo)
print(MergedInfo.isnull().values.any())
print(MergedInfo.nunique())
#put rearrange to logical order
print(MergedInfo.columns.values)
MergedInfo = MergedInfo[['src', 'content_building', 'content_flower', 'content_bicycle', 'content_people', 'content_dog', 'content_cars', 'content_cat', 'content_tree', 'content_bird',
                            'emotion_happy', 'emotion_scary', 'emotion_gloomy', 'emotion_peaceful', 'media_comic', 'media_3d_graphics', 'media_vectorart', 'media_graphite', 'media_pen_ink', 
                            'media_oilpaint', 'media_watercolor']]#Gets only used columns and orders them so that it is consistent
print(MergedInfo.columns.values)
#change negative to 0, unsure to .5, and positive to 1
MergedInfo = MergedInfo.replace('negative', 0).replace('unsure', .5).replace('positive', 1)
print(MergedInfo)
#save to csv for GetData.py to download the images from and store the tags
MergedInfo.to_csv('../dataset/info.csv', index=False)