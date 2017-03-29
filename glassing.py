import numpy as np 
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

df=pd.read_csv('datasets/glass.csv')
# print (df.describe())
X=df.iloc[:,:8]
y=df.iloc[:,9]

#KNN
classifier=KNeighborsClassifier(n_neighbors=1)
print (cross_val_score(classifier,X,y,cv=5).sum()/5)




