import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('datasets/LinearRegression5_Data.csv')
print(data)

print(data.shape)    #(6,2)

X = data.iloc[ : , 0:1].values    # [ rows , cols ]
y = data.iloc[:, 1].values

print("X.shape = ", X.shape , "\n X=\n" , X)

print("y.shape = ", y.shape , "\n y=" , y )


from sklearn.linear_model import LinearRegression

lin = LinearRegression()
lin.fit(X, y)
y_dash = lin.predict(X)

plt.scatter(X, y, color='blue')
plt.plot(X, y_dash , color='red')
plt.title('Linear Regression')
plt.xlabel('Engine Temperature')
plt.ylabel('Engine Pressure')

plt.show()
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

#poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)


plt.scatter(X, y, color='blue')

plt.plot(X, lin2.predict(X_poly), color='red')
plt.title('Polynomial Regression')
plt.xlabel('Engine Temperature')
plt.ylabel('Engine Pressure')

plt.show()





# Predicting a new result with Linear Regression
print( "LinearRegresion: ", lin.predict([[110.0]])  )

# Predicting a new result with Polynomial Regression
print( "PolynomialRegresion: ",lin2.predict(poly.fit_transform([[110.0]]))   )