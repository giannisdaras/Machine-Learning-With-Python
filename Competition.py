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

df=pd.read_csv('datasets/glass.csv')
print (df.describe())
X=df.iloc[:,0:8] #choose correct columns
y=df.iloc[:,9] #choose correct columns
fold_numbers=5 #constant for kfold

#Information about the dataset
targets=[]
for i in y:
	if (i not in targets):
		targets.append(i)
times=[]
yList=list(y)
for j in targets:
	times.append("%.2f" %(yList.count(j)/len(yList)))
print ("Targets: ",targets)
print ("Total number of samples: ",len(y))
print ("Percentage of each class: ",times)

#Feature extraction:
# pcaExtractor=PCA(n_components=5)
# X=pcaExtractor.fit_transform(X)

# Selecting features
# X=SelectKBest(chi2,k=3).fit_transform(X,y)

# Data standarization
# sc=StandardScaler()
# X=sc.fit_transform(X)

#Data normalization	
# X=preprocessing.scale(X)

#KNN
knnSimpleClf=KNeighborsClassifier(n_neighbors=3)
# k=np.arange(0,80,1)+1
# parameters={'n_neighbors':k}
# clf=GridSearchCV(knnSimpleClf,parameters,cv=5)
# clf.fit(X,y)
# print (clf.best_params_)
print ('Knn scoring: ',cross_val_score(knnSimpleClf,X,y,cv=fold_numbers).sum()/fold_numbers)

# svm=SVC()
# Co=(0.01,1.0,10.0,100.0)
# kerns=('linear','rbf','poly')
# gammas=(0.01,1.0,10.0,100.0)
# params={'C':Co,'kernel':kerns,'gamma':gammas}
# clf=GridSearchCV(svm,params,cv=5)
# clf.fit(X,y)
# print (clf.best_params_)
svm=SVC(kernel='linear',C=100.0,gamma=1.0)
print ('SVM score: ',cross_val_score(svm,X,y,cv=fold_numbers).sum()/fold_numbers)

#Forest
# criteria=('entropy','gini')
# estimators=np.arange(0,100,10)+1
# params={'criterion':criteria,'n_estimators':estimators}
# forest=RandomForestClassifier()
# clf=GridSearchCV(forest,params,cv=5)
# clf.fit(X,y)
# print (clf.best_params_)
forest=RandomForestClassifier(criterion='entropy',n_estimators=20,n_jobs=1)
print ('Forest scoring: ',cross_val_score(forest,X,y,cv=fold_numbers).sum()/fold_numbers)

# Logistic Regression
# logr=LogisticRegression()
# penalize=('l1','l2',)
# Co=(0.01,1.0,10.0,100.0)
# params={'C':Co,'penalty':penalize}
# clf=GridSearchCV(logr,params,cv=5)
# clf.fit(X,y)
# print (clf.best_params_)
logr=LogisticRegression(penalty='l2',C=10)
print ('Logistic scoring: ',cross_val_score(logr,X,y,cv=fold_numbers).sum()/fold_numbers)

#MLP
# learning_rates=(0.001,0.01,0.1)
# hidden_layers=((100,),(500,),(1000,))
# params={'learning_rate_init':learning_rates,'hidden_layer_sizes':hidden_layers}
# multiP=MLPClassifier()
# clf=GridSearchCV(multiP,params,cv=5)
# clf.fit(X,y)
# print (clf.best_params_)
multiP=MLPClassifier(learning_rate_init=0.01,hidden_layer_sizes=(500,))
print ('MLP scoring: ',cross_val_score(logr,X,y,cv=fold_numbers).sum()/fold_numbers)

#Naive Bayes
bayesian=GaussianNB()
print ('Bayesian naive scoring: ',cross_val_score(logr,X,y,cv=fold_numbers).sum()/fold_numbers)
