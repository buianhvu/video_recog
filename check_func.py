# import numpy as np

# a = np.ones((4,4), int)
# b = [5,8,7,1] #d=4
# print("b checks")
# print b[0:3]


# print a
# np.fill_diagonal(a, b)
# print a
# print np.diag(a)

# print("reg")
# reg = np.identity(4+1)
# reg[4,4] = 6
# print
import numpy as np
from sklearn.svm import SVC

# yy = np.ones((3,))
# yy_arr = []
# for y in yy:
# 	yy_arr.append(str(y))
# print (type(yy[1]))
# print (type(yy_arr[1]))
# print yy_arr
# print yy
# print("yy shape {}".format(yy.shape))

x_train = np.random.rand(3,5)
y_train = ["a","b","c"]
xx_test = np.random.rand(3,5)
yy_test = ["a","b","c"]

clf = SVC(gamma='auto')
clf.fit(x_train, y_train)

score = clf.score(xx_test, yy_test)
print ("Score {}".format(score))
