import numpy as np
import h5py 

dir =  "data/"
list_data_names = ["llc_iDTs_cam0.mat","llc_iDTs_cam1.mat","llc_iDTs_cam2.mat","llc_iDTs_cam3.mat","llc_iDTs_cam4.mat"]
list_label_names = ["llc_label_cam0.mat", "llc_label_cam1.mat", "llc_label_cam2.mat", "llc_label_cam3.mat", "llc_label_cam4.mat"]



def load_data(dir, list_file_names):
	i = 0
	x = [[] for n in range(5)]
	for filename in list_file_names:
		f = h5py.File(dir+filename)
		print(f.keys)
		print ("Reading...: {}".format(filename))
		print("length file: {}".format(len(f["llc_iDTs"])))
		for element in f["llc_iDTs"]:
			# print("element shape: {}".format(element.shape))
			x[i].append(element)
		xx_i = np.array(x[i])
		np.save("xx_{}".format(i), xx_i)
		i = i + 1
	# x = np.array(x)
	# print("X shape: {}".format(x.shape))
	pass

def load_labels(dir, list_label_names):
	i = 0
	x = [[] for n in range(5)]
	for filename in list_label_names:
		f = h5py.File(dir+filename)
		print ("Reading...: {}".format(filename))
		for element in f["llc_label"]:
			x[i].append(element)
		i = i + 1
	x = np.array(x)
	np.save("yy_np", x)
	pass

load_data(dir,list_data_names)
load_labels(dir, list_label_names)
