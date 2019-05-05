import sam_lib as slib
import numpy as np
from sklearn.svm import SVC
print("loading yy...")
yy = np.load("vectors/yy_np.npy")
print("loading xx...")

xx0 = np.load("vectors/xx_0.npy").transpose()
print(xx0.shape)
xx1 = np.load("vectors/xx_1.npy").transpose()
xx2 = np.load("vectors/xx_2.npy").transpose()
xx3 = np.load("vectors/xx_3.npy").transpose()
xx4 = np.load("vectors/xx_4.npy").transpose()

xx = np.concatenate((xx0,xx1,xx2,xx3),axis = 1) #xx dxn
yy0,yy1,yy2,yy3,yy4 = yy[0,:], yy[1,:], yy[2,:], yy[3,:], yy[4,:]
yy = np.concatenate((yy0, yy1, yy2, yy3), axis = 1)
# yy0,yy1,yy2,yy3,yy4 = yy[0,:], 
print("done loading")


# print(xx.shape)
# print(xx0.shape)
# print(yy.shape)
# print(yy0.shape)


print("Data input shape {}".format(xx.shape))
Z = slib.cal_Z(xx, 4)
print("Finish cal Z")
Ws, hx = slib.msda_z(xx,Z,0.6,5)
print("shape hx {}".format(hx.shape))
x_clf = hx.transpose()
clf = SVC(gamma='auto')
yy = yy.reshape(yy.shape[1],)
clf.fit(x_clf, yy)
x_test_clf = xx4.transpose()
y_test_clf = yy4.reshape(yy4.shape[1],)
score = clf.score(x_test_clf, y_test_clf)
print("Score train on , test on X1: {}".format(score))

