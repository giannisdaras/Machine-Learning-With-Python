import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from math import exp
from sklearn import datasets

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
		result=[]
		probabilities=[]
		for i in data:
			nooby=np.dot(i,self.W[1:])+self.W[0]
			probabilities.append(1/(1+exp(-nooby)))
			if nooby>0:
				nooby=1
			else:
				nooby=-1
			result.append(nooby)
		return result,probabilities
df = datasets.load_iris()
y = df.target[0:100]
y = np.where(y == 0, -1, 1)
X = df.data[0:100,[0,2]]
fancyAdaline=Adaline(0.01,15)
fancyAdaline.fit(X,y)
results,probabilities=fancyAdaline.predict(X)
print "misclassified: ", (y!=results).sum()
for i in range(len(probabilities)):
	if abs(probabilities[i]-0.5)<0.1:
		print "Dangerous position ", i , "with probability ", probabilities[i]



