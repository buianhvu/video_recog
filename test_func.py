import numpy as np
import numpy.matlib as mb
import math

# mat = np.ones((5,5))
# # print mat
# # np.fill_diagonal(mat, np.zeros((1,5))) 
# print mat

# mat1 = np.random.rand(5,6)
# mat2 = mat1[0:4,:]
# print mat1
# print mat2

# mat3 = [1,3,4]
# mat4 = mb.repmat(mat3, 2,3)
# print mat3
# print mat4.shape

mat5 = np.array([4, 4, 4])
mat6 = np.array([2,2,2])

print math.exp(math.sqrt(np.sum(np.square(mat5-mat6))))