# Save Model Using Pickle
import warnings
warnings.filterwarnings(action='ignore')
import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle


filename = 'datasets/indians-diabetes.data.csv'
headingnames=['preg', 'plas', 'pres', 'skin', 'test',
              'mass', 'pedi', 'age', 'class']

dataframe = pd.read_csv(filename, names=headingnames)


array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                        test_size=0.33, random_state=seed)

# Fit the model on 67% data
model = LogisticRegression()
model.fit(X_train, Y_train)

# save the model to disk
filename =  "finalized_model.sav"
pickle.dump(model, open(filename,  "wb" ))
#wb-> write binary; loads the data.. readable only by machine

print( "Model dumped successfully into a file by Pickle"
       "....\n...\n...\n...")

print("-----------------------\n\n\n")
print("some time later...  ")
print("\n\n\n-----------------------")
# load the model from disk
loaded_model = pickle.load(open(filename,  "rb" ))
#rb-> read binary; reads the data.. readable only by machine
print( "Model loaded successfully from file by Pickle")

result = loaded_model.score(X_test, Y_test)
print( "Accuracy Result : " , result)
