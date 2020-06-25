import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

filename = 'datasets/housing.csv'

hnames=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

dataframe = pd.read_csv(filename, names=hnames)
array = dataframe.values

X = array[: , 0:13]
Y = array[:,13]

kfold = KFold(n_splits=10 )
model = LinearRegression()

scoringMethod = 'neg_mean_absolute_error' #MAE
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoringMethod)

print( "MAE: %.3f (%.3f)" % ( results.mean(), results.std() ) ) 

# Attribute Information:
# ---------------------------------------------------------------------------
# 1. CRIM: per capita crime rate by town
# 2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
# 3. INDUS: proportion of non-retail business acres per town
# 4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# 5. NOX: nitric oxides concentration (parts per 10 million)
# 6. RM: average number of rooms per dwelling
# 7. AGE: proportion of owner-occupied units built prior to 1940
# 8. DIS: weighted distances to five Boston employment centres
# 9. RAD: index of accessibility to radial highways
# 10. TAX: full-value property-tax rate per 10,000 US Dollars
# 11. PTRATIO: pupil-teacher ratio by town
# 12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# 13. LSTAT: % lower status of the population
# 14. MEDV: Median value of owner-occupied homes in 1000's US Dollars 