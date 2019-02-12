from sklearn import preprocessing
import numpy as np

np.random.seed(0)
X = np.random.randint(10,size=(5,3))
print(X)

scaled_X = preprocessing.scale(X)
print(scaled_X)

print("Mean before scaling" ,X.mean(axis=0))

print("Mean after scaling" ,scaled_X.mean(axis=0))

print("Std deviation before scaling" ,X.std(axis=0))

print("Std deviation after scaling" ,scaled_X.std(axis=0)) 





binarizer = preprocessing.Binarizer()
binarized_X = binarizer.transform(X)
print("binarized_X:")
print(binarized_X)


print("With Threshold:")
binarizer = preprocessing.Binarizer(threshold= 5)
binarized_X = binarizer.transform(X)
print(binarized_X)





normalizer = preprocessing.Normalizer()
normalized_X = normalizer.transform(X)
print(normalized_X)



imputer = preprocessing.Imputer()
print(imputer)
imputer.fit([[1,2],[3,4],[5,np.nan]])
X = np.array([[np.nan,6],[4, np.nan],[2,3]])
imputed_X = imputer.transform(X)
print(imputed_X)

import pandas as pd
data = {"location": ["beach", "highway","city"],"temp": [10,25,30]}

df =pd.DataFrame(data, index=['loc1', 'loc2', 'loc3'])
print("\noriginal Data Frame:\n",df)

labelEncoder= preprocessing.LabelEncoder()
df['location']=labelEncoder.fit_transform(df['location'])

print("\nAfter Lable Encoding on location:\n",df)
hotEncoder = preprocessing.OneHotEncoder(categorical_features=[0])
encoded_df= hotEncoder.fit_transform(df).toarray()
print("\nAfter One Hot Encoding:\n",encoded_df)