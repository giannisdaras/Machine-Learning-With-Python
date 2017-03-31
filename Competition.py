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

df=pd.read_csv('datasets/Iris.csv')
# print (df.describe())
X=df.iloc[:,1:4]
y=df.iloc[:,5]
fold_numbers=5

X=SelectKBest(chi2,k=3).fit_transform(X,y)
# sc=StandardScaler()
# X=sc.fit_transform(X)	
# X=preprocessing.scale(X)

#KNN
knnSimpleClf=KNeighborsClassifier(n_neighbors=9)
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
svm=SVC(kernel='poly',C=0.01,gamma=1.0)
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
multiP=MLPClassifier(learning_rate_init=0.01,hidden_layer_sizes=(100,))
print ('MLP scoring: ',cross_val_score(logr,X,y,cv=fold_numbers).sum()/fold_numbers)

#Naive Bayes
bayesian=GaussianNB()
print ('Bayesian naive scoring: ',cross_val_score(logr,X,y,cv=fold_numbers).sum()/fold_numbers)
