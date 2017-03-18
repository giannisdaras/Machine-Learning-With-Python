import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import datasets
from sklearn.cross_validation import train_test_split

class Knn:
	def __init__(self,K):
		self.K=K
		return
	def predict(self,instance,dataset,target_values):
		euclideanDistances=[]
		for i in range(dataset.shape[0]):
			euclideanDistances.append((math.sqrt(sum((instance-dataset[i])**2)),target_values[i]))
		euclideanDistances.sort(key=lambda euclideanDistances: euclideanDistances[0])
		result=0
		for j in range(self.K):
			if euclideanDistances[j][1]==1:
				result+=1
		if (result>self.K/2):
			return 1
		else:
			return 0
flowers=datasets.load_iris()
X=flowers.data[0:100,[0,2]]
y=flowers.target[0:100]
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=0)
classifierKnn=Knn(10)
predictions=[]
for i in X_test:
	predictions.append(classifierKnn.predict(i,X_train,Y_train))
print ((Y_test-predictions)!=0).sum()



