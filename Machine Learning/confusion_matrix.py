import warnings
warnings.simplefilter(action="ignore", category=Warning)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

filename = 'datasets/indians-diabetes.data.csv'
names=['preg', 'plas', 'pres', 'skin', 'test',
                'mass', 'pedi', 'age', 'class']

dataframe = pd.read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

test_size = 0.33
#seed = 7
#,random_state=seed

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=test_size )

model = LogisticRegression()
model.fit(X_train, Y_train)  # m , c

predicted = model.predict(X_test)  # y --> sigmoid


matrix = confusion_matrix(Y_test, predicted)

print("confusion_matrix is \n",matrix)