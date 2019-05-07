import numpy as np
import math
import scipy.linalg
import time

def init_G(views, d):
	#each Gv has shaped dx(d+1) with bias:
	G = []
	for view in range(views):
		Gv = np.random.rand(d, d+1)
		G.append(Gv)
	return G


#Z VnxVn
def cal_Z(xx,V):
	#x has shape [x1,x2,x3,...] it is good for cal G
	
	# print("in cal_Z xx shape {}".format(xx.shape))

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
					diff = xx[:,k+i*n]-xx[:,k+j*n]
					K[i, j, k] = math.exp(math.sqrt(np.sum(np.square(diff))))/(2*2)
			
	# Z = np.zeros((V*n, V*n))
	Z = K[:,:,0]
	for i in range(n):
		if i != 0:
			Z = scipy.linalg.block_diag(Z, K[:,:,i])
	print("Finish calculating Z {}".format(Z.shape))
	return Z
	pass



def compute_gg_inve(G, beta, _lambda):
	dim = G[0].shape[0] #dim = d
	print("dim gv = {}".format(dim))
	reg = np.identity(dim)
	x = np.zeros((dim,dim))
	for g in G:
		# print("X {} g {} reg: {}".format(x.shape, g.dot(g.transpose()).shape, reg.shape))
		x = x + g.dot(g.transpose())*beta + reg
	x_inve = np.linalg.inv(x)
	return x # x has shape dxd

def mda_z(xx, gg, Z, noise, lambda_, alpha, beta, V, Converge):
	#xx is [xx1, xx2, xx3, xx4]
	#V views => there are V elements in G
	GG = [] #each element of GG is vth-x views
	for x_v in gg:
		# print("x_v shapeee {}".format(x_v.shape))
		bias = np.ones((1, x_v.shape[1]))
		x_v_bias = np.concatenate((x_v, bias),axis = 0)
		GG.append(x_v_bias)
	print("GG is added bias")

	d, N = xx.shape
	# print("in mda_z xx shape {}".format(xx.shape))
	# print ("aaaaaaaaaaaaa")
	n = N/V
	G = init_G(V, d)

	#adding bias
	b_mt = np.ones((1,N))
	xxb = np.concatenate((xx,b_mt), axis = 0)
	xxz = xx.dot(Z)
	#adding bias to xxz
	xxz = np.concatenate((xxz,b_mt), axis = 0)
	S = xxb.dot(xxb.transpose())
	Sz = xxz.dot(xxz.transpose())
	print("SZ shape {}".format(Sz.shape))
	# print Sz
	# print Sz[0:d,:]
	#corruption vector
	q = np.ones((d+1,1))*(1-noise)
	q[-1] = 1
	#Q d+1 x d+1
	Q = np.multiply(S, q.dot(q.transpose()))
	np.fill_diagonal(Q,np.multiply(q,np.diag(S)))
	#P dx(d+1) P = Sz(1:end-1,:).*repmat(q', d, 1);
	# print("Sz[0:d,:] shape {}".format(Sz[0:d,:].shape))
	# print(" np.tile(q.transpose(), (d, 1)) shape {}".format( np.tile(q.transpose(), (d, 1)).shape))
	P = np.multiply(Sz[0:d,:], np.tile(q.transpose(), (d, 1)))
	#final W = P*Q^-1, dx(d+1)
	reg = np.identity(d+1)
	# print("reg shape {}".format(reg.shape))
	reg[d,d] = 0
	#W dx(d+1)
	del S; del xxz; del Z;
	M = P.dot(np.linalg.inv(Q+reg))
	del Q; del P;

	#some pre-data for computing Gv:
	# SG = [v for v in range(V)]
	# QG = [v for v in range(V)]
	# PG = [v for v in range(V)]
	# for view in range(V):
	# 	print('view: {}'.format(view))
	# 	print('check point 2:')
	# 	SG[view] = GG[view].dot(GG[view].transpose()) #each has shape d+1 x d+1
	# 	print('check point 2.0')
	# 	QG[view] = np.multiply(SG[view], q.dot(q.transpose())) #shape d+1 x d+1
	# 	print('check point 2.1')
	# 	np.fill_diagonal(QG[view], np.multiply(q,np.diag(SG[view]))) #d+1 x d+1
	# 	print('check point 2.2')
	# 	PG[view] = np.multiply(SG[view][0:d,:], np.tile(q.transpose(),(d,1))) #dx(d+1)
	# 	print('check point 2.2.2:')
	print('check point 0:')
	id_mat = np.identity(d)
	print("Shape id_mat {}".format(type(id_mat)))
	#tills converges
	print("Converging")
	for converge in range(Converge):
		print("Converge : {}".format(converge))
		W = compute_gg_inve(G, beta, lambda_).dot(M) #update W
		print("W type {} Wshape {}".format(type(W), W.shape))
		W_to_G = beta*W.dot(W.transpose())+id_mat
		inve_W_to_G = np.linalg.inv(W_to_G) #dxd
		#after updating W, then update Gv:
		for view in range(V):
			print("Updating G{}".format(view))
			SG = GG[view].dot(GG[view].transpose())		
			print("check point 1")
			QG = np.multiply(SG, q.dot(q.transpose())) #shape d+1 x d+1
			print("check point 2")
			PG = np.multiply(SG[0:d,:], np.tile(q.transpose(),(d,1))) #dx(d+1)
			print("check point 3")
			del SG
			print("check point 3.0")
			temp = (alpha*PG).dot(np.linalg.inv(alpha*QG+reg)) #dx(d+1)
			print("check point 3.1")
			del QG
			del PG
			print("check point 4")
			#update G[view]
			G[view] = inve_W_to_G.dot(temp) #dx(d+1)
			print("End updating G{}".format(view))

	print("Converging done")
	# print("W shape {}".format(W.shape))
	hw = W.dot(xxb)
	hw = np.tanh(hw)

	hg = [n for n in range(V)]
	for view in range(V):
		hg_ = G[view].dot(GG[view])
		hg[view] = np.tanh(hg_)

	return hw, hg, W, G
	pass



def msda_z(xx, gg, Z, noise, layers, lambda_, alpha, beta, V, Converge):
	#xx : dxn input
	#noise: corruption level
	#layers: number of layers to stack
	#allhx: (layers*d)xn stacked hidden representations
	print("**************STACKING HIDDEN LAYERS")
	print("Input Shape: {}".format(xx.shape))
	print("Noise: {}, Layers: {}, Lambda: {}".format(noise, layers, lambda_))
	prevhw = xx
	prevhg = gg
	allw = []
	allg = []
	Ws = []
	Gs = []
	for layer in range(layers):
		print("**Layer number: {}".format(layer+1))
		time1 = time.time()
		new_hw, new_hg, new_W, new_G = mda_z(prevhw,prevhg,Z,noise,lambda_, alpha,beta,V)
		time2 = time.time()-time1
		print("Run in time: {}".format(time2))
		Ws.append(new_W)
		Gs.append(new_G)
		prevhw = new_hw
		prevhg = new_hg
	return new_W, new_G
	pass



# mda_z(fake_xx, fake_z, 0.8, 1)



# xx_ = np.random.rand(3, 25)
# xx_ = np.array(xx_)
# Z = cal_Z(xx_, 5)
# print("**********START")
# msda_z(xx_,Z,0.6,4,1)