import csv
filename= open('datasets/indians-diabetes.data.csv', 'r')
reader=csv.reader(filename, delimiter=',')

lines = list(reader)
print("\n\nNo of rows:", len(lines), "\n\n")

# print( lines )

print("\n\n")

for l in lines :
	print( l )
