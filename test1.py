import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

df=pd.read_csv('datasets/dataset1.csv')
X=df.iloc[:,0:93] #choose correct columns
y=df.iloc[:,94] #choose correct columns

sc=StandardScaler()
X=sc.fit_transform(X)
fold_numbers=5 #constant for kfold
# X=SelectKBest(k=80).fit_transform(X,y)

# multiP=MLPClassifier(learning_rate_init=0.001,hidden_layer_sizes=(1000,))
# # multiP.fit(X,y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# multiP.fit(X_train,y_train)
# multiPredict=multiP.predict(X_test)
# f1=f1_score(y_test,multiPredict)
# print (f1)

# df=pd.read_csv('datasets/testVectors1.csv',header=None)
# X=df.iloc[:,0:93] #choose correct columns
# X=sc.transform(X)
# multiPredict=multiP.predict(X)
# np.savetxt("Try1.csv", multiPredict, fmt='%d')

logr=LogisticRegression(penalty='l1',C=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
logr.fit(X_train,y_train)
logrPredict=logr.predict(X_test)
f1=f1_score(y_test,logrPredict)
print (f1)
df=pd.read_csv('datasets/testVectors1.csv',header=None)
X=df.iloc[:,0:93] #choose correct columns
X=sc.transform(X)
logrPredict=logr.predict(X)
np.savetxt("Try1.csv", logrPredict, fmt='%d')

# forest=RandomForestClassifier(criterion='entropy',n_estimators=31,n_jobs=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# forest.fit(X_train,y_train)
# forestPredict=forest.predict(X_test)
# f1=f1_score(y_test,forestPredict)
# print (f1)
# df=pd.read_csv('datasets/testVectors1.csv',header=None)
# X=df.iloc[:,0:93] #choose correct columns
# X=sc.transform(X)
# forestPredict=forest.predict(X)
# np.savetxt("Try1.csv", forestPredict, fmt='%d')






