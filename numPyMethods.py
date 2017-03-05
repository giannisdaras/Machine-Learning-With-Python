import numpy as np
from numpy import pi 
# How to create a numpy array object:
# np.array(list of tuples)
# example:
a=np.array([(1,2,3),(200,20,9)])
##print a

#Methods: np.zeros(3,4) np.ones(5,6) , np.empty(100,9)
b=np.arange(10,40,10) #it gives values from 10 to 40 using step10.
##print b
c=np.linspace(0,3,9) #it gives 9 evenly spaced numbers between 0 and 3.
##print c

a=np.linspace(0,100,12)
a=a.reshape(3,4)  #if the second parameter is -1, then is automatically computed
##print a

#Basic Operations
a=np.array([200,100,300])
a=a+1 #adds to every element 1
#print (a**2)

c=np.array([100,200,300])
d=np.array([100,200,300])
#print (c-d)
#print (c<200)
print (np.vstack((c,d))) #vazei ton enan pinaka katw apo ton allon
print (np.hstack((c,d))) #vazei ton enan pinaka dipla apo ton allon

#Products
A_matrix=np.array([(100,200),(500,800)])
B_matrix=np.array([(1,2),(3,4)])
#print (A_matrix*B_matrix) #element (to element) product
#print (A_matrix.dot(B_matrix)) #matrix product
#print (A_matrix.sum()) 
#print (A_matrix.max())
#print (A_matrix.min())
#print (A_matrix.sum(axis=0)) #sum of each column
#print (A_matrix.sum(axis=1)) #sum of each row


print ("iterations begin")
#Please note that!
rand_matrix=np.array([(100,200,300),(1,2,3)])
#ways to iterate!
for i in rand_matrix:
	print i
for i in rand_matrix.flat:
	print i
for i in rand_matrix:
	for j in i:
		print j

#Changing the shape of the array
machine_Learnign_Arr=np.array([(200,300,400),(600,700,1903),(3,6,9)])
print machine_Learnign_Arr.ravel() #kanei ton pinaka 1xm.
print machine_Learnign_Arr.T