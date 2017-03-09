import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

class Adaline:
	def __init__(self,learning_rate,epochs):
		self.learning_rate=learning_rate
		self.epochs=epochs
	def fit(self,X,y):
		self.W=np.zeros(1+X.shape[1])
		
		for l in range(self.epochs):
			y_=[]
			for t in range(X.shape[0]):
				predicted=self.W[0] + np.dot(X[t,:],self.W[1:])
				y_.append(predicted)
			#weight updates:
			sum0=0
			for iterat in range(X.shape[0]):
				sum0=sum0 + y[iterat]-y_[iterat]
			update_const=sum0*self.learning_rate
			self.W[0]+=update_const
			DW=[]
			for j in range(X.shape[1]):
				sum1=0
				for iterator in range(X.shape[0]):
					sum1=sum1+ (y[iterator]-y_[iterator])*X[iterator,j]
				DW.append([self.learning_rate*sum1])
			for it in range(X.shape[1]):
				self.W[it+1]+=DW[it]
		return;
	def predict(self,data):
		output=self.W[0] + np.dot(data,self.W[1:])
		print ("output value:", output)
		if (output<0) :
			return 'Iris-setosa'
		else:
			return 'Iris-vertosa'

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y=df.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',-1,1)
X=df.iloc[0:100,[0,2]].values
fancyAdaline=Adaline(0.01,50)
fancyAdaline.fit(X,y)
returned=fancyAdaline.predict(df.iloc[52,[0,2]].values)
# print ("Predicted: ",returned)
# print ('Actual: ', df.iloc[51,4])
# for i in range(150):
# 	print ("Actual ", df.iloc[i,4])
# 	print ("Predicted ",fancyAdaline.predict(df.iloc[i,[0,2]].values))










