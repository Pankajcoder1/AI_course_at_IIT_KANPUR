import numpy as np
import matplotlib.pyplot as plt

def estimate_coef(x, y):
	# number of observations/points
	n = np.size(x)

	# mean of x and y vector
	m_x = np.mean(x)
	m_y = np.mean(y)

	# calculating cross-deviation and deviation about x
	SS_xy = np.sum(x * y) - n * m_x * m_y
	SS_xx = np.sum(x * x) - n * m_x * m_x

	# calculating regression coefficients
	m = SS_xy / SS_xx
	c = m_y - m * m_x
	return [m, c]

def plot_regression_line(x, y, b):
# plotting the actual points as scatter plot
	plt.scatter(x, y, color="m", marker="o", s=30)

	# predicted response vector
	y_pred = b[1] * x + b[0]

	# plotting the regression line
	plt.scatter(x, y_pred, color="g")
	plt.plot(x, y_pred, color="b")

	# putting labels
	plt.xlabel('----x---->')
	plt.ylabel('----y---->')
	# function to show plot
	plt.show() 

def startAIAlgorithm():
# observations
	x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ] )
	y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

	# estimating coefficients
	m , c = estimate_coef(x, y) #Sender Values
	print( "Estimated coefficients:\n ")
	print( "slope m = ", m)
	print( "y-intercept c =" , c)
	print("y = ", m , " * x + " , c)
	# plotting regression line
	plot_regression_line(x, y, [c,m])

if __name__ == "__main__":
	startAIAlgorithm() 