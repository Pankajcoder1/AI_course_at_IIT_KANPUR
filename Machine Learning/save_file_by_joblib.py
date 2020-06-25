# Save Model Using joblib
#More efficient than pickle
import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

import warnings
warnings.filterwarnings(action='ignore')
filename = 'datasets/indians-diabetes.data.csv'
names=['preg', 'plas', 'pres', 'skin', 'test',
       'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                     test_size=0.33, random_state=7)
# Fit the model on 67% data
model = LogisticRegression()
model.fit(X_train, Y_train)

# save the model to disk
filename =  "finalized_model.sav"
joblib.dump(model, filename)
print( "Model dumped successfully into a file by Joblib")
print("\n....\n...\n...\n...")

# some time later...
print("-----------------------\n\n\n")
print("some time later...  ")
print("\n\n\n-----------------------")


# load the model from disk
loaded_model = joblib.load(filename)
print( "Model loaded successfully from file by Joblib")
result = loaded_model.score(X_test, Y_test)
print( "Accuracy Result : " , result)


#these are used to create a file in which the trained model is loaded
# so that it can be used in future by various users.