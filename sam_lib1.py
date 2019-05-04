import numpy as np
import math
import scipy.linalg
# x1 = np.random.rand(5)
# x2 = np.random.rand(5)
# x3 = np.random.rand(5)
# V = 4

# # print x1
# # print x2
# # print x3
# fake_xx = [x1, x2, x3]
# fake_xx = np.array(fake_xx)
# fake_xx = fake_xx.transpose()
# print("Fake_xx {}".format(fake_xx))
# fake_d, fake_n = fake_xx.shape
# fake_z = np.random.rand(fake_n, fake_n)
# print("fake_Z shape {}".format(fake_z.shape))
# #Z VnxVn
def cal_Z(xx,V):
	#n != N
	#V domian size
	d, N = xx.shape
	print ("d = {} N = {}".format(d,N))
	n = N/V
	K = np.zeros((V,V,n))
	for k in range(n):
		for i in range(V):
			for j in range(V):
				# print np.square(xx[:,k+n*(i-1)]-xx[:,k+n(j-1)])
				print("point check 1")
				print (xx[:, k+n*i])
				print (xx[:, k+n*j])
				if i == j:
					K[i, j, k] = 0
				else:
					K[i, j, k] = math.exp(-math.sqrt(np.sum(np.square(xx[:,k+n*i]-xx[:,k+n*j]), dtype = np.double)))/(2*2)
			
	# Z = np.zeros((V*n, V*n))
	Z = np.zeros((0,0))
	for i in range(n):
		Z = scipy.linalg.block_diag(Z, K[:,:,i])
	print Z.shape
	return Z
	pass

def mda_z(xx, Z, noise, _lambda):
	d, n = xx.shape
	# print ("n {} r {}".format(n, d))
	#adding bias
	b_mt = np.ones((1,n))
	# print b_mt
	# print("bias shape {}".format(b_mt.shape))
	xxb = np.concatenate((xx,b_mt), axis = 0)
	# print("xxb: {}".format(xxb))
	# print ("xxb shape {}".format(xxb.shape))

	xxz = xx.dot(Z)
	# print ("xx shape {}".format(xx.shape))
	# print ("xxz shape {}".format(xxz.shape))
	# print ("bias vector shape {}".format(b_mt.shape))
	#adding bias to xxz
	xxz = np.concatenate((xxz,b_mt), axis = 0)
	print("xxz_b shape {}".format(xxz.shape))
	S = xxb.dot(xxb.transpose())
	print ("S shape {}".format(S.shape))

	Sz = xxz.dot(xxz.transpose())
	print("Sz shape {}".format(Sz.shape))
	print("Sz {}".format(Sz))
	#corruption vector
	q = np.ones((d+1,1))*(1-noise)
	q[-1] = 1
	#Q d+1 x d+1
	Q = np.multiply(S, q.dot(q.transpose()))
	np.fill_diagonal(Q,np.multiply(q,np.diag(S)))
	print("q {}".format(q))
	print("Q {}".format(Q))
	print("Q shape {}".format(Q.shape))
	#P dx(d+1) P = Sz(1:end-1,:).*repmat(q', d, 1);
	# print(q.transpose())
	# print(np.tile(q.transpose(), (d, 1)))
	P = np.multiply(Sz[0:d,:], np.tile(q.transpose(), (d, 1)))
	#final W = P*Q^-1, dx(d+1)
	reg = _lambda*np.identity(d+1)
	print("reg shape {}".format(reg.shape))
	reg[d,d] = 0
	#W dx(d+1)
	W = P.dot((Q+reg).transpose())
	print("W shape {}".format(W.shape))
	hx = W.dot(xxb)
	hx = np.tanh(hx)
	return W, hx
	pass
def msda_z(xx, Z, noise, layers, lambda_):

	#xx : dxn input
	#noise: corruption level
	#layers: number of layers to stack
	#allhx: (layers*d)xn stacked hidden representations
	print("**************STACKING HIDDEN LAYERS")
	prevhx = xx
	allhx = []
	Ws = []
	for layer in range(layers):
		print("**Layer number: {}".format(layer+1))
		new_W, new_hx = mda_z(xx,Z,noise,lambda_)
		Ws.append(new_W)
		prevhx = new_hx
	return Ws
	pass


# mda_z(fake_xx, fake_z, 0.8, 1)



xx_ = np.random.rand(3, 25)
xx_ = np.array(xx_)
Z = cal_Z(xx_, 5)
print("**********START")
msda_z(xx_,Z,0.6,4,1)