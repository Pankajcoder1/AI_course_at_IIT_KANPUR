	# Developed by Pankaj Kumar.
	# country: India
	# class assignment.


# the task for this assignment is the draw the best fit line and 
# best fit curve to minimize error for LR_Episodes datasets

try:

	import warnings
	warnings.simplefilter(action="ignore", category=Warning)
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import train_test_split
	from sklearn.model_selection import KFold
	from sklearn.model_selection import cross_val_score
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.linear_model import LinearRegression
	from sklearn.preprocessing import PolynomialFeatures


	datas=pd.read_csv('datasets/LR_Episodes.csv')
	print("some rows of data is \n",datas.head())
	print("shape of data is : ",datas.shape)    #(6,4)

	# datas.hist()
	# plt.show()
	# extract data from file 

	X1=datas.iloc[:,0:1]
	Y1=datas.iloc[:,1]
	print("\nX1 is ",X1)
	print("\nY1 is ",Y1)
	X2=datas.iloc[:,2:3].values
	Y2=datas.iloc[:,1].values

	print("\nX2 is ",X2)
	print("\nY2 is ",Y2)
	print("\nX1.shape is= ",X1.shape)
	print("\nY1.shape is= ",Y1.shape)


	model=LinearRegression()
	model.fit(X2,Y2)
	Y_res=model.predict(X2)
	plt.title("Plotting of linear_model")
	plt.plot(X2,Y_res,color='yellow')
	plt.scatter(X2,Y2,color='red')
	plt.xlabel("X2 values")
	plt.ylabel("Y2 values")
	plt.show()

	# here degree of PolynomialFeatures is 
	# chosen by hit and trail method.(apply and then check)
	poly=PolynomialFeatures(degree=8)
	X_ploy=poly.fit_transform(X2)
	model=LinearRegression()
	model.fit(X_ploy,Y2)
	plt.title("Plotting of polynomial_model")
	plt.xlabel("X2 values")
	plt.ylabel("Y2 values")
	plt.scatter(X2,Y2,color='red')
	Y_res2=model.predict(X_ploy)
	plt.plot(X2,Y_res2)
	plt.show()
	Y_res2=model.score(X_ploy,Y2)
	print(f"accuracy is {Y_res2*100}")

except:
	print("First install all required module. ")