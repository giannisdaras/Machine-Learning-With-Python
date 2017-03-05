import numpy as np
import pandas as pd #Library for testing our perceptron 
import matplotlib.pyplot as plt
class Perceptron():
	def __init__(self,learning_rate,epochs):
		self.learning_rate=learning_rate
		self.epochs=epochs 
		#epochs: number of iterations through the training set	
	def fit(self,X,y):
		#X numpy array for input
		#X: NxM
		#N samples
		#M features
		#y numpy vector for output
		#y: Nx1
		#W numpy vector for weights + threshold
		#W: (M+1)x1
		#y_ the estimated output value
		self.W=np.zeros(1+X.shape[1]) 
		for i in range(self.epochs):
			for xi,target in zip(X,y):
				y_=self.W[0]+np.dot(xi,self.W[1:])
				#weight updates!
				update=self.learning_rate*(target-y_)
				print update
				self.W[0]+=update
				self.W[1:]+=update*xi
		

	def predict(self,data):
		output=self.W[0] + np.dot(data,self.W[1:])
		return output
	
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y=df.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',-1,1)
X=df.iloc[0:100,[0,2]].values
plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')
plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='versicolor')
plt.xlabel('Petal length')
plt.ylabel('Sepal length')
plt.legend(loc='upper left')
#plt.show()

fancyPerceptron=Perceptron(0.1,10)
fancyPerceptron.fit(X,y)

