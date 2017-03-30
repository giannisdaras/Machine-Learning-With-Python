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

df=pd.read_csv('datasets/Iris.csv')
# print (df.describe())
X=df.iloc[:,1:4]
y=df.iloc[:,5]
fold_numbers=5

# X=SelectKBest(chi2,k=7).fit_transform(X,y)
# sc=StandardScaler()
# X=sc.fit_transform(X)	
# X=preprocessing.scale(X)

#KNN
knnSimpleClf=KNeighborsClassifier(n_neighbors=3)
# parameters={'n_neighbors':list(range(100))}
clf=GridSearchCV(knnSimpleClf,parameters,cv=5)
clf.fit(X,y)
print (clf.best_params_)
# print ('Knn scoring: ',cross_val_score(knnSimpleClf,X,y,cv=fold_numbers).sum()/fold_numbers)


#SVM
svm=SVC(kernel='linear',C=1)
print ('SVM scoring: ',cross_val_score(svm,X,y,cv=fold_numbers).sum()/fold_numbers)

#Gaussian kernel
svmGaussian=SVC(kernel='rbf',gamma=1.0,C=1.0)
print ('Gaussian kernel scoring: ',cross_val_score(svmGaussian,X,y,cv=fold_numbers).sum()/fold_numbers)

#SVM poly
svmPoly=SVC(kernel='poly',C=1.0)
print ('SvmPoly scoring: ',cross_val_score(svmPoly,X,y,cv=fold_numbers).sum()/fold_numbers)

#Forest
forest=RandomForestClassifier(criterion='entropy',n_estimators=100,random_state=1,n_jobs=1)
print ('Forest scoring: ',cross_val_score(forest,X,y,cv=fold_numbers).sum()/fold_numbers)

#Logistic Regression
logr=LogisticRegression(penalty='l1',C=100)
print ('Logistic scoring: ',cross_val_score(logr,X,y,cv=fold_numbers).sum()/fold_numbers)

#MLP
multiP=MLPClassifier(hidden_layer_sizes=(800,))
print ('MLP scoring: ',cross_val_score(logr,X,y,cv=fold_numbers).sum()/fold_numbers)

