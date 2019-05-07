import sam_lib as slib
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
d = 2000
n = 489
xx0 = np.load("cam0.npy").reshape(2000, 489)
xx1 = np.load("cam1.npy").reshape(2000, 489)
xx2 = np.load("cam2.npy").reshape(2000, 489)
xx3 = np.load("cam3.npy").reshape(2000, 489)
xx4 = np.load("cam4.npy").reshape(2000, 489)

yy0 = np.load("label0.npy")
yy1 = np.load("label1.npy")
yy2 = np.load("label2.npy")
yy3 = np.load("label3.npy")
yy4 = np.load("label4.npy")


for i in range(n):
	xx0[:,i] = xx0[:,i]/np.sum(xx0[:,i])
	xx1[:,i] = xx1[:,i]/np.sum(xx1[:,i])
	xx2[:,i] = xx2[:,i]/np.sum(xx2[:,i])
	xx3[:,i] = xx3[:,i]/np.sum(xx3[:,i])
	xx4[:,i] = xx4[:,i]/np.sum(xx4[:,i])

xx = np.concatenate((xx0,xx1,xx2,xx3,xx4),axis = 1) #xx dxN
print("xx1 shape: {}".format(xx1.shape))
print("xx shape: {}".format(xx.shape))
yy = [yy0,yy1,yy2,yy3,yy4] #1xN


Z = slib.cal_Z(xx)
#msda_z(xx, gg, Z, noise, layers, lambda_, alpha, beta, V):
# from sklearn.svm import SVC

print("msda_z starts ....")
Ws, Gs = slib.msda_z(xx, Z, 0.6, 1)
W = Ws[-1]

np.save("W_np",W)
# np.save("G_np", G)

print("Finish msda_Z")

#print("TESTING")
# print("hw type: {}".format(hw))
del Z
#starting get accuracy
print("Initializing classifier: ")
# clf = SVC(gamma='auto')
clf = KNeighborsClassifier(n_neighbors=1)
#multi-to-one if x2 is choose for test, then it is excluded from the training
bias_train = np.ones((1,n))
x_train = np.concatenate((xx1,bias_train), axis = 0)
x_train = (W.dot(x_train)).transpose()
print("X_TRAIN SHAPE: {}".format(x_train.shape))
y_train = yy1
print("Y_TRAIN SHAPE: {}".format(y_train.shape))
# xx = np.concatenate((xx,np.ones((1,xx.shape[1]))), axis = 0)
# yy_train_arr = []
# for y in y_train:
# 	int_y = int(y)
# 	yy_train_arr.append(int_y)
# xx = np.concatenate((xx,np.ones((1,xx.shape[1]))), axis = 0)

# print ("Y train arr: {}".format(yy_train_arr))
clf.fit(x_train, y_train)

biases = np.ones((1, n))
#supposed tested on xx2
x_test = np.concatenate((xx2,biases), axis=0) #(d+1)xn
x_test = W.dot(x_test) #(d+1, n)
x_test = x_test.transpose() #feed to svm

y_test = yy2
print ("Y_TEST  = {}".format(y_test))
print("y_test type {}".format(type(y_test[1])))
# yy_test_arr = []
# for yt in y_test:
# 	int_yt = int(yt)
# 	yy_test_arr.append(int_yt)

predict = clf.predict(x_test)
print("predict: {}".format(predict))
print("Y test arr: {}".format(y_test))
score = clf.score(x_test, yy_test_arr)
print("calculating score ...")
print("Score train on , test on X2: {}".format(score))

