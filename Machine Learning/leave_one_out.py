import warnings
warnings.filterwarnings(action="ignore")
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

filename = 'datasets/indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test',
'mass', 'pedi', 'age', 'class']

dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

loocv = LeaveOneOut()
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=loocv)

print( "results : " , results )

print( "result.size : " , results.size )

print( "Sum of Positive Results: %i " % (
results.sum() ) )

print( "Accuracy= %.2f %%" % (
results.sum() * 100 / results.size ) ) 