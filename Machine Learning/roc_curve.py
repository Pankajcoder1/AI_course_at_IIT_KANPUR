import warnings
warnings.filterwarnings(action="ignore")


from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
filename = 'indians-diabetes.data.csv'
names=['preg', 'plas', 'pres', 'skin', 'test',
       'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)

array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10 )

model = LogisticRegression()

scoringMethod = 'roc_auc'

results = cross_val_score(
    model, X, Y, cv=kfold, scoring=scoringMethod)

print("AUC: %.3f (%.3f)" % (
    results.mean()*100, results.std()*100 ) )