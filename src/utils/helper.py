"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Loading data
"""

import os, shutil
import numpy as np
import scipy
import torch

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
		
class Helper:

	__device__ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	@staticmethod
	def deactivate(model):
		for p in model.parameters():
			p.requires_grad = False

	@staticmethod
	def activate(model):
		for p in model.parameters():
			p.requires_grad = True

	@staticmethod
	def sample_z(m, n):
		return np.random.normal(size=[m, n], loc = 0, scale = 1)
		# return np.random.uniform(-1., 1., size=[m, n])

	@staticmethod
	def save_sample(samples, sample_path, label):
		# save npy
		filepath = '{}/samples_{}.npy'.format(sample_path, label)
		np.save(filepath, samples.cpu().data.numpy())

	@staticmethod
	def weights_init(m):
		classname = m.__class__.__name__
		if classname.find('Linear') != -1:
			m.weight.data.normal_(0.0, 0.02)
			if m.bias is not None:
				m.bias.data.fill_(0)
		elif classname.find('Conv') != -1:
			torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
		elif classname.find('BatchNorm') != -1:
			m.weight.data.normal_(1.0, 0.02)
			# m.bias.data.fill_(0)
			m.bias.data.normal_(0, 0.02)

	@staticmethod
	def sparse_mx_to_torch_sparse_tensor(sparse_mx):
		"""Convert a scipy sparse matrix to a torch sparse tensor."""
		sparse_mx = sparse_mx.tocoo().astype(np.float32)
		indices = torch.from_numpy(
			np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
		values = torch.from_numpy(sparse_mx.data)
		shape = torch.Size(sparse_mx.shape)
		return torch.sparse.FloatTensor(indices, values, shape)

	@staticmethod
	def normalize(mx):
		"""Row-normalize sparse matrix"""
		rowsum = np.array(mx.sum(1))
		r_inv = np.power(rowsum, -1).flatten()
		r_inv[np.isinf(r_inv)] = 0.
		r_mat_inv = sp.diags(r_inv)
		mx = r_mat_inv.dot(mx)
		return mx

	################### Ops ##################
	@staticmethod
	def mkdir(name, rm=False):
		if not os.path.exists(name):
			os.makedirs(name)

	@staticmethod
	def log(logf, msg, console_print=True):
		logf.write(msg + '\n')
		if console_print:
			print(msg)

	@staticmethod
	def save_mat(name, pvalues):
		scipy.io.savemat(name, {'pvalues':pvalues})
