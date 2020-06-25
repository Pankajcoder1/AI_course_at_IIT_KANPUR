import warnings
warnings.filterwarnings(action="ignore")

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

filename = 'datasets/indians-diabetes.data.csv'
headingNames = ['preg', 'plas', 'pres',
                'skin', 'test', 'mass',
                'pedi', 'age', 'class']

dataframe = pd.read_csv(filename, names=headingNames)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
print(X)
print("tpes os X2 is ",type(X))
kfold = KFold(n_splits=10)

#1) Spot cheching for Logistic Regression
#---------------------------------------------------------
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold )
print( "Validation Score for LogisticRegression : " , results.mean()  )


#2) Spot cheching for Linear Discriminant Analysis(LDA)
#---------------------------------------------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis()
results = cross_val_score(model, X, Y, cv=kfold)
print("Validation Score for Linear Discriminant Analysis:", results.mean()  )



#3) Spot cheching for k-Nearest Neighbors (kNN)
#---------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

results = cross_val_score(model, X, Y, cv=kfold)

print(  "Validation Score for kNN : " , results.mean()  )