import pandas as pd
import warnings
warnings.filterwarnings(action="ignore")
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut,cross_val_score,KFold,ShuffleSplit


filename='datasets/indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test',
			'mass', 'pedi', 'age', 'class']
data=pd.read_csv(filename,names=names)
# print(data.head())
X=data.iloc[:,0:8]
Y=data.iloc[:,8]
X=X.values
Y=Y.values
data.hist()
plt.show()


model=LogisticRegression()
shuffle=ShuffleSplit(n_splits=10,test_size=0.33)
res=cross_val_score(model,X,Y,cv=shuffle)
print(f"accuracy by repeated train test is {res.mean()*100} %")