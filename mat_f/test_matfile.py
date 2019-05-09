import numpy as np
import h5py 
# f = h5py.File('msda_fea.mat')
# print(f.keys)
import scipy.io
mat = scipy.io.loadmat('z.mat')
a = mat['Z']
print np.save('Z', a)