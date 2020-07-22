"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Loading data
"""
import os
import numpy as np
import scipy.io as spio
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from utils.helper import Helper
import matplotlib.pyplot as plt


class Provider:

	@staticmethod
	# Dataset iterator
	def inf_train_gen(data, batch_size):
		ds_size = data.shape[0]
		while True:
			for i in range(ds_size // batch_size):
				start = i * batch_size
				end = (i + 1) * batch_size
				yield data[start:end, :]

	@staticmethod
	def load_data(dataset_path, dataset_name):
		data_path = os.path.join(dataset_path, '{}/data_adni.npy'.format(dataset_name))
		print(data_path)
		dict = np.load(data_path, allow_pickle=True).item()
		# signals = np.asarray(dict[mode_datatype])
		ads = np.asarray(dict['ad_signals'])
		cns = np.asarray(dict['cn_signals'])
		A = dict['A']
		L = dict['L']
		return ads, cns, A, L

	@staticmethod
	def generate_data(dataset_path, dataset_name, extra=0, sparse=False):
		in_data_path = os.path.join(dataset_path, '{}/data_adni.mat'.format(dataset_name))
		print(in_data_path)
		out_data_path = os.path.join(dataset_path, '{}/data_adni.npy'.format(dataset_name))
		dict = spio.loadmat(in_data_path, squeeze_me=True)
		ads = np.asarray(dict['ad_signal'])
		cns = np.asarray(dict['cn_signal'])
		cns = np.asarray([e for e in cns[:]])
		ads = np.asarray([e for e in ads[:]])
		A = np.array(dict['A'], dtype=np.float)

		print("AD count: {:d}".format(len(ads)))
		print("CN count: {:d}".format(len(cns)))

		mu_ad = np.mean(ads, axis=0)
		std_ad = np.std(ads, axis=0)

		dropped_vtx = np.where(std_ad < 0.01)
		#print("Dropping vertices {} due to small variation".format(dropped_vtx.tolist()))

		mask = np.zeros(ads.shape[1], dtype=np.bool)
		mask[dropped_vtx] = True
		ads = ads[:, ~mask]
		cns = cns[:, ~mask]
		A = A[~mask, :][:, ~mask]

		# print(A)
		# A = normalize(A + sp.eye(A.shape[0]))
		L = sp.csgraph.laplacian(A, normed=True)
		if sparse:
			A = Helper.sparse_mx_to_torch_sparse_tensor(A)
			L = L.todense()
	
		
		# args.signal_size -= np.size(dropped_vtx)

		n_plots = 3 if extra > 0 else 2
		plt.subplot(1, n_plots, 1)
		plt.hist(ads.flatten(), bins=20)
		plt.title("AD stats")
		plt.subplot(1, n_plots, 2)
		plt.hist(cns.flatten(), bins=20)
		plt.title("CN stats")

		if extra > 0:
			print("Augmented Samples: {:d}".format(extra))
			mu = np.mean(cns, axis=0)
			cov = np.cov(cns.transpose())
			extra_data = np.random.multivariate_normal(mu, cov, extra)
			cns = np.concatenate([cns, extra_data], axis=0)

			plt.subplot(1, 3, 3)
			plt.hist(np.mean(extra_data, axis=0), bins=20)
			plt.title("CN Augmented Stats")
		plt.savefig("../debug/data_stats.png")

		data = {'ad_signals': ads, 'cn_signals': cns, 'L':L, 'A':A}
		np.save(out_data_path, data)
