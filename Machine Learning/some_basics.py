import numpy as np

k = 3

max = 120

C_x = np.random.randint(0, max-20, size=k)
print("C_x = ",  C_x)



C_y = np.random.randint(0, max-20, size=k)
print("C_y = ",   C_y)

C = list( zip( C_x, C_y)  )
C = np.array(C)
print("C = \n" , C)

print( "C.shape = ",  C.shape)

C_old =   np.zeros( C.shape  )
print("C_old = \n", C_old )

ar5 = np.array([22,44,11,66,55])

print( "ar5 = ", ar5)

print( np.argmin( ar5 ) )

#list Comprehension

ar1 = [11,12,13,14,15,16,17,18,19,20]
print( ar1 )

ar2 = list( range(11,21) )
print( ar2 )


for num in ar1 :
    if(num%2 == 0) :
        print( num , end=" ")

ar3 = [ n for n in ar1 ]
print("\n\nar3 = " , ar3)

ar4 = [ n for n in ar1 if(n%2 == 0) ]   #list comprehension
print("\n\nar4 = " , ar4)

