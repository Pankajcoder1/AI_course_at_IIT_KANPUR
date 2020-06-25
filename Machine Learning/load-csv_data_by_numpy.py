import numpy

filename = 'datasets/indians-diabetes.data.csv'
raw_data = open(filename, 'r')

data = numpy.loadtxt(raw_data, delimiter=",")

print( "Numpy loadtxt : " ,data.shape ) #(768,9)
raw_data.close()

print("\n\n\n")

print( data )


print("-*-" * 30 )

# for l in data :
# 	print( l ) 