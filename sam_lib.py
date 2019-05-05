import numpy as np
import math
import scipy.linalg
import time


#Z VnxVn
def cal_Z(xx,V):
	#n != N
	#V domian size
	d, N = xx.shape
	n = N/V
	K = np.zeros((V,V,n))
	for k in range(n):
		for i in range(V):
			for j in range(V):
				# print np.square(xx[:,k+n*(i-1)]-xx[:,k+n(j-1)])
				if i == j:
					K[i, j, k] = 0
				else:
					K[i, j, k] = math.exp(-math.sqrt(np.sum(np.square(xx[:,k+n*i]-xx[:,k+n*j]))))/(2*2)
			
	# Z = np.zeros((V*n, V*n))
	Z = K[:,:,0]
	for i in range(n):
		if i != 0:
			Z = scipy.linalg.block_diag(Z, K[:,:,i])
	print(Z.shape)
	return Z
	pass

def mda_z(xx, Z, noise, _lambda):
	d, n = xx.shape
	#adding bias
	b_mt = np.ones((1,n))
	xxb = np.concatenate((xx,b_mt), axis = 0)
	xxz = xx.dot(Z)
	#adding bias to xxz
	xxz = np.concatenate((xxz,b_mt), axis = 0)
	S = xxb.dot(xxb.transpose())
	Sz = xxz.dot(xxz.transpose())
	print("SZ shape {}".format(Sz.shape))
	print Sz
	print Sz[0:d,:]
	#corruption vector
	q = np.ones((d+1,1))*(1-noise)
	q[-1] = 1
	#Q d+1 x d+1
	Q = np.multiply(S, q.dot(q.transpose()))
	np.fill_diagonal(Q,np.multiply(q,np.diag(S)))
	#P dx(d+1) P = Sz(1:end-1,:).*repmat(q', d, 1);
	print("Sz[0:d,:] shape {}".format(Sz[0:d,:].shape))
	print(" np.tile(q.transpose(), (d, 1)) shape {}".format( np.tile(q.transpose(), (d, 1)).shape))
	P = np.multiply(Sz[0:d,:], np.tile(q.transpose(), (d, 1)))
	#final W = P*Q^-1, dx(d+1)
	reg = _lambda*np.identity(d+1)
	# print("reg shape {}".format(reg.shape))
	reg[d,d] = 0
	#W dx(d+1)
	W = P.dot((Q+reg).transpose())
	# print("W shape {}".format(W.shape))
	hx = W.dot(xxb)
	hx = np.tanh(hx)
	return W, hx
	pass
def msda_z(xx, Z, noise, layers):
	lambda_ = 0.00001
	#xx : dxn input
	#noise: corruption level
	#layers: number of layers to stack
	#allhx: (layers*d)xn stacked hidden representations
	print("**************STACKING HIDDEN LAYERS")
	print("Input Shape: {}".format(xx.shape))
	print("Noise: {}, Layers: {}, Lambda: {}".format(noise, layers, lambda_))
	prevhx = xx
	allhx = []
	Ws = []
	for layer in range(layers):
		print("**Layer number: {}".format(layer+1))
		time1 = time.time()
		new_W, new_hx = mda_z(prevhx,Z,noise,lambda_)
		time2 = time.time()-time1
		print("Run in time: {}".format(time2))
		Ws.append(new_W)
		prevhx = new_hx
	return Ws, new_hx
	pass


# mda_z(fake_xx, fake_z, 0.8, 1)



# xx_ = np.random.rand(3, 25)
# xx_ = np.array(xx_)
# Z = cal_Z(xx_, 5)
# print("**********START")
# msda_z(xx_,Z,0.6,4,1)