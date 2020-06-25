# 1) Cross Validation Classification Accuracy
import warnings
warnings.filterwarnings(action="ignore")
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

filename = 'datasets/indians-diabetes.data.csv'
names=['preg', 'plas', 'pres', 'skin',
'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)

array = dataframe.values
X = array[: , 0:8]
Y = array[:,8]
kfold = KFold(n_splits=10 )
model = LogisticRegression()

# This is the default method of accuracy
scoringMethod = 'accuracy'


results = cross_val_score( model, X, Y,cv=kfold , scoring=scoringMethod )

print( "Accuracy: %.3f (%.3f)" % ( results.mean()*100,
results.std()*100 ) ) 