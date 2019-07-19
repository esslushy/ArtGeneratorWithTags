import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load Data
df = pd.read_csv('../dataset/all_data_info.csv')
# Get Important Columns
df = df[['genre', 'style', 'new_filename']]
# Drop Nulls and Invalid
df = df.dropna()
# One Hot Encode the columns
labelEncoder = LabelEncoder()
oneHot = OneHotEncoder(sparse=False)
# One Hot Encoding
genreOneHot = oneHot.fit_transform(df['genre'].values.reshape(-1, 1))
np.savetxt('../utils/genreCategories.txt', oneHot.categories_, fmt='%s', encoding='utf-8', delimiter='\n')
styleOneHot = oneHot.fit_transform(df['style'].values.reshape(-1, 1))
np.savetxt('../utils/styleCategories.txt', oneHot.categories_, fmt='%s', encoding='utf-8', delimiter='\n')
# Save Image Locations and labels
np.save('../dataset/imageLocation.npy', df['new_filename'].values)
# Append labels together
labels = np.append(genreOneHot, styleOneHot, axis=1)
np.save('../dataset/labels.npy', labels)