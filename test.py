import sam_lib_mod as slib
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

# xx1 = np.random.rand(3,5)
# xx2 = np.random.rand(3,5)
# xx3 = np.random.rand(3,5)
# xx4 = np.random.rand(3,5)
gg = [xx0,xx1,xx2,xx3]
gg = np.array(gg)



xx = xx1
print("xx shape: {}".format(xx1.shape))


Z = slib.cal_Z(xx,1)
#msda_z(xx, gg, Z, noise, layers, lambda_, alpha, beta, V):
from sklearn.svm import SVC
print("msda_z ....")


W, G, hw, hg = slib.msda_z(xx, gg, Z, 0.6, 1, 1, 1, 1, 1)
print("hw type: {}".format(hw))
print("Initializing classifier: ")
x_clf = hw.transpose()
clf = SVC(gamma='auto')
yy = yy.reshape(yy.shape[1],)
clf.fit(x_clf, yy)

biases = np.ones((1, 330))
print(xx4.shape)
print(biases.shape)
xx4 = np.concatenate((xx4,biases), axis=0)
xx4 = W.dot(xx4)
x_test_clf = xx4.transpose()
y_test_clf = yy4.reshape(yy4.shape[1],)
score = clf.score(x_test_clf, y_test_clf)
print("calculating score ...")
print("Score train on , test on X4: {}".format(score))



# # yy0,yy1,yy2,yy3,yy4 = yy[0,:], 
# print("done loading")

