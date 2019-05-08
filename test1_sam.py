import sam_lib as slib
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
import sys
print("loading yy...")
yy = np.load("vectors/yy_np.npy")
print("loading xx...")

xx0 = np.load("vectors/xx_0.npy").transpose()
xx1 = np.load("vectors/xx_1.npy").transpose()
xx2 = np.load("vectors/xx_2.npy").transpose()
xx3 = np.load("vectors/xx_3.npy").transpose()
xx4 = np.load("vectors/xx_4.npy").transpose()

n = 330 #test on smaller set for speeding up
xx0 = xx0[:,:n] #dxn
xx1 = xx1[:,:n]
xx2 = xx2[:,:n]
xx3 = xx3[:,:n]
xx4 = xx4[:,:n]

#concatenate
xx = np.concatenate((xx0,xx1,xx2,xx3,xx4),axis = 1) #xx dxN
print("xx1 shape: {}".format(xx1.shape))
print("xx shape: {}".format(xx.shape))

yy0,yy1,yy2,yy3,yy4 = yy[0,:], yy[1,:], yy[2,:], yy[3,:], yy[4,:] #1xn
#test on smaller set for speeding up
yy0 = yy0[:,:n] #1xn
yy1 = yy1[:,:n]
yy2 = yy2[:,:n]
yy3 = yy3[:,:n]
yy4 = yy4[:,:n]

print("YY0 : {}".format(yy0))
print("YY1 : {}".format(yy1))
print("YY2 : {}".format(yy2))
print("YY3 : {}".format(yy3))
print("yy0 shape {}".format(yy0.shape))
yy = np.concatenate((yy0, yy1, yy2, yy3, yy4), axis = 1) #1xN
print("yy shape: {}".format(yy.shape))
gg = [xx0,xx1,xx2,xx3,xx4]
gg = np.array(gg)
V = 5
print("gg shape: {}".format(gg.shape))

Z = slib.cal_Z(xx)
#msda_z(xx, gg, Z, noise, layers, lambda_, alpha, beta, V):
# from sklearn.svm import SVC

print (Z)
# sys.exit(1)
print("msda_z starts ....")
Ws, allhx = slib.msda_z(xx, Z, 0.6, 1)
W = Ws[-1]
hx = allhx[-1]
hx_xx = np.concatenate((xx,hx))
print("hx_xx shape: {}".format(hx_xx.shape))
print("hx shape: {}".format(hx.shape))
print("W shape: {}".format(W.shape))
# np.save("G_np", G)

print("Finish msda_Z")

#print("TESTING")
# print("hw type: {}".format(hw))
del Z
#starting get accuracy
print("Initializing classifier: ")
# clf = SVC(gamma='auto')
clf = SVC()

x_train = hx_xx[:,0:1*n].transpose()
# yy_train = np.array([yy1,yy2])
# y_train = yy_train.reshape(660,).astype(str)
y_train = yy1.reshape(yy1.shape[1],).astype(int)


clf.fit(x_train, y_train)

x_test = hx_xx[:,n:2*n].transpose()
y_test = yy1.reshape(yy1.shape[1],).astype(int)

predict = clf.predict(x_test)
score = clf.score(x_test, y_test)
print("PREDICT: {}".format(predict))
print("SCORE : {}".format(score))