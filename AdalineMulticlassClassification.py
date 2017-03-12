import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

class Adaline(object):
	def __init__(self,learning_rate,epochs):
		self.learning_rate=learning_rate
		self.epochs=epochs
		return;
	def fit(self,X,y):
		self.W=np.zeros(1+X.shape[1])
		for k in range(self.epochs):
			X,y=self.shuffler(X,y)
			for i in range(X.shape[0]):
				y_=self.W[0]+ np.dot(X[i,:],self.W[1:])
				update=(y[i]-y_)*self.learning_rate
				for l in range(X.shape[1]):
					self.W[l+1]+=update*X[i,l]
				self.W[0]+=update
		return;
	def shuffler(self,X,y):
		r=np.random.permutation(len(y))
		return X[r],y[r]
	def predict(self,data):
		return np.dot(data,self.W[1:])+self.W[0]

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
X = df.iloc[0:150, [0, 2]].values
y = df.iloc[0:150, 4].values
Y1 = np.where(y == 'Iris-setosa', -1, 1)
Y2=np.where(y=='Iris-versicolor',-1,1)
Y3=np.where(y=='Iris-virginica',-1,1)
fancyAdaline1=Adaline(0.01,15)
fancyAdaline1.fit(X,Y1)
fancyAdaline2=Adaline(0.01,15)
fancyAdaline2.fit(X,Y2)
fancyAdaline3=Adaline(0.01,15)
fancyAdaline3.fit(X,Y3)
results=[]
results.append(fancyAdaline1.predict(df.iloc[140,[0,2]].values))
results.append(fancyAdaline2.predict(df.iloc[140,[0,2]].values))
results.append(fancyAdaline3.predict(df.iloc[140,[0,2]].values))
minIndex=results.index(min(results))
print minIndex #0: Iris-setosa, 1:Iris-versicolor, 2:Iris-virginica
print results
