import sam_lib as slib
import numpy as np
from sklearn.svm import SVC
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

print("msda_z starts ....")
Ws, Gs = slib.msda_z(xx, Z, 0.9, 1)
W = Ws[-1]

np.save("W_np",W)
# np.save("G_np", G)

print("Finish msda_Z")

#print("TESTING")
# print("hw type: {}".format(hw))
del Z
#starting get accuracy
print("Initializing classifier: ")
clf = SVC(gamma='auto')
#multi-to-one if x2 is choose for test, then it is excluded from the training
bias_train = np.ones((1,n))
x_train = np.concatenate((xx1,bias_train), axis = 0)
x_train = (W.dot(x_train)).transpose()
print("X_TRAIN SHAPE: {}".format(x_train.shape))
y_train = yy1.reshape(yy1.shape[1],)
print("Y_TRAIN SHAPE: {}".format(y_train.shape))
# xx = np.concatenate((xx,np.ones((1,xx.shape[1]))), axis = 0)

clf.fit(x_train, y_train)

biases = np.ones((1, n))
#supposed tested on xx2
x_test = np.concatenate((xx2,biases), axis=0) #(d+1)xn
x_test = W.dot(x_test) #(d+1, n)
x_test = x_test.transpose() #feed to svm
y_test = yy2.reshape(yy2.shape[1],)

score = clf.score(x_test, y_test)
print("calculating score ...")
print("Score train on , test on X2: {}".format(score))



# # # yy0,yy1,yy2,yy3,yy4 = yy[0,:], 
# print("done loading")

