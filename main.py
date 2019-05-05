import sam_lib as slib
import numpy as np
print("loading yy...")
yy = np.load("vectors/yy_np.npy")
print("loading xx...")

xx0 = np.load("vectors/xx_0.npy").transpose()
xx1 = np.load("vectors/xx_1.npy").transpose()
xx2 = np.load("vectors/xx_2.npy").transpose()
xx3 = np.load("vectors/xx_3.npy").transpose()
xx4 = np.load("vectors/xx_4.npy").transpose()

xx = np.concatenate((xx0,xx1,xx2,xx3,xx4),axis = 1) #xx dxn
yy0,yy1,yy2,yy3,yy4 = yy[0,:], yy[1,:], yy[2,:], yy[3,:], yy[4,:]
yy = np.concatenate((yy0, yy1, yy2, yy3, yy4), axis = 1)
# yy0,yy1,yy2,yy3,yy4 = yy[0,:], 
print("done loading")

# print(xx.shape)
# print(xx0.shape)
# print(yy.shape)
# print(yy0.shape)

Z = slib.cal_Z(xx0, 1)
Ws = slib.msda_z(xx0,Z,0.6,4,1)

print(Z.shape)
print(Ws.shape)