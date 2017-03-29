import numpy as np 
import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('datasets/glass.csv')
print (df.describe())
print (df.iloc[:,9].values)

#KNN
classifier=KNeighborsClassifier(n_neighbors=1)
z=0
for i in range(0,100):
	X_train,X_test,y_train,y_test=train_test_split(df.iloc[:,:8],df.iloc[:,9])
	# stdsc = StandardScaler()
	# X_train=stdsc.fit_transform(X_train)
	# X_test=stdsc.fit_transform(X_test)
	# X_train=preprocessing.scale(X_train)
	# X_test=preprocessing.scale(X_test)
	classifier.fit(X_train,y_train)
	y_pred=classifier.predict(X_test)
	z+=(y_test!=y_pred).sum()
print 'Knn algorithm: (n=3)' ,z/101





