import numpy as np

a = np.ones((4,4), int)
b = [5,8,7,1] #d=4
print("b checks")
print b[0:3]


print a
np.fill_diagonal(a, b)
print a
print np.diag(a)

print("reg")
reg = np.identity(4+1)
reg[4,4] = 6
print reg