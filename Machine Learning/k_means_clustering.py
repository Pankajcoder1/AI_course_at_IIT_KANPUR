from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings(action = "ignore")
#plt.style.use('ggplot')

#Manual calculation
#plt.rcParams['figure.figsize'] = (8,6)
# Importing the dataset
data = pd.read_csv('datasets/KMeansData.csv')
print("Input Data and Shape")
print(data.shape)   # (3000, 2)
print( data.head() )

# Getting the values and plotting it
f1 = data['V1'].values
f2 = data['V2'].values

plt.scatter(f1, f2, c='black', s=10)
plt.show()

X = np.array( list( zip(f1, f2))  )
#[[2.072345,-3.241693], [], []]

print(X)

#  a = [12 ,700]
#  b = [5, 300 ]
# -----------

# Euclidean Distance Caculator
def eu_dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


# Number of clusters
k = 3
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(X), size=k)
print( "C_x = " , C_x )
# Y coordinates of random centroids
C_y = np.random.randint(0, np.max(X), size=k)
print( "C_y = " , C_y )

C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print("Initial Centroids (Random Position) : ")
print(C)
print(  C.shape   )   # (3,2)

# Plotting along with the Centroids
plt.scatter(f1, f2,  s=10, c='k' )
plt.scatter(C_x, C_y, marker='*', s=300, c='r')
plt.show()


# To store the value of centroids when it updates
C_old = np.zeros(C.shape)   #C.shape = [3,2]
print( "C =  \n", C )
print( "C_old \n" , C_old )

print( 'len(X) = ' , len(X) )


# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))



#zero filled numpy array of 3000 elements
print( "clusters : = " , clusters )



# Error func. - Distance between new centroids
# and old centroids
error = eu_dist(C, C_old)

print("Error before loop", error)
# Loop will run till the error becomes zero
while error.all():      # error != 0
    # Assigning each value to its closest cluster
    for i in range(len(X)):          # len(X) : 3000
        distances = eu_dist(X[i], C)      # distances = [ 12 , 50 , 9 ]
        cluster = np.argmin(distances) # cluster = 2
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for s in range(k):  # k=3  because we have to find 3 centroid locaton
        points = [  X[j] for j in range(len(X)) if clusters[j] == s ]
        C[s] = np.mean(points, axis=0)
    error = eu_dist(C, C_old)
    print("Error in Loop", error)

colors = ['b', 'c', 'r']


fig, ax = plt.subplots()
for i in range(k):    # k=3
    points=np.array([ X[j] for j in range(len(X)) if clusters[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s=25, c=colors[i])

ax.scatter(C[:, 0], C[:, 1], marker='*', s=100, c='y')

print( "Final Centroid : " , C )
plt.show()


#Using core machine learning

from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
# Importing the dataset
data = pd.read_csv('datasets/KMeansData.csv')
print( "KMeans of sklearn"  )
# Number of clusters
kmeans = KMeans(n_clusters=3)

f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(  list( zip(f1, f2) )   )

# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
print( "labels : " , labels )
print( list( zip(X, labels) )  )

# Centroid values
centroids = kmeans.cluster_centers_
print("Best place for new Shop\n", centroids)


# Comparing with scikit-learn centroids
print( "KMeans Algo Centroid values :- \n")

print( "KMeans:Manual centroid\n" , C)

print( "KMeans sklearn:centroids \n",centroids )  # From scikit-learn
