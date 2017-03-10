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
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
# X_std=np.copy(X)
# X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
# X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
fancyAdaline=Adaline(0.01,15)
fancyAdaline.fit(X,y)

for i in range(100):
	print "Actual ", df.iloc[i,4]
 	if (fancyAdaline.predict(df.iloc[i,[0,2]].values)>=0):
 		print "Predicted: Iris-versicolor"
 	else:
 		print "Predicted: Iris-setosa"
