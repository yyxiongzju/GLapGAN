"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Loading data
"""

import os, sys, time
sys.path.append('..')
import numpy as np
# import sklearn
from sklearn.metrics import recall_score, precision_score

import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from configs import args
from utils.helper import Helper, AverageMeter
from utils.provider import Provider
from utils.ttester import TTester

################## Setting #######################
##### Model ######
MODELS = ['lapgan', 'baseline', 'real']
DATASETS = ['adni', 'adrc', 'simuln', 'synthetic']
DATATYPES = ['ad', 'cn']

## Data path
dataset_name = DATASETS[args.mode_dataset]
mode_datatype = DATATYPES[args.mode_datatype]

## Output path
model_name = MODELS[args.mode_model] + '-' + dataset_name
output_path = os.path.join(args.result_path, model_name)
# saved checkpoint
model_path = os.path.join(output_path, 'snapshots-{}'.format(args.num_ads))
netG_path = os.path.join(model_path, 'netG.pth')
netD_path = os.path.join(model_path, 'netD.pth')
# samples
sample_path = os.path.join(output_path, 'samples')
log_path = os.path.join(output_path, "log.txt")

# makedir
Helper.mkdir(args.result_path)
Helper.mkdir(output_path)
Helper.mkdir(model_path)
Helper.mkdir(sample_path)
logf = open(log_path, 'w')

#####====================== Data ================######
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
device = args.gpu
ADs, CNs, A, L = Provider.load_data(args.dataset_path, dataset_name)

# import matplotlib.pyplot as plt
# bins = np.linspace(0, 5, 100)
# plt.subplot(121)
# plt.hist(ADs.flatten(), bins=bins)
# plt.subplot(122)
# plt.hist(CNs.flatten(), bins=bins)
# plt.show()

if args.mode_dataset == 1:
	A = Helper.sparse_mx_to_torch_sparse_tensor(A)
	L = L.todense()
# elif not(args.mode_dataset == 3):
# 	ADs = (ADs + 5)/15
# 	CNs = (CNs + 5)/15

L = torch.tensor(L, dtype=torch.float32).to(device)
A = torch.tensor(A, dtype=torch.float32).to(device)

# data_mean = np.mean(CNs)
# data_stddev = np.std(CNs)
# scaling_fn = lambda x: (x - data_mean) / data_stddev
# rescaling_fn = lambda y: y * data_stddev + data_mean

nsamples = args.num_cns
#num_training_samples = int(args.train_fraction * CNs.shape[0])
num_training_samples = args.num_cns
# CNs_ttest = CNs[:nsamples, :]
# cn_data = CNs[nsamples:nsamples+num_training_samples, :]
# CNs_ttest = CNs[:num_training_samples]
# cn_data = CNs[:num_training_samples]
CNs_ttest = CNs[:nsamples]
cn_data = CNs[:nsamples]

# normalize data into [-1, 1]
range_cns = [np.min(cn_data) - 0.5, np.max(cn_data) + 0.5]
print("Data range: {}".format(range_cns))
scaling_fn = lambda x: (x - range_cns[0]) / (range_cns[1] - range_cns[0]) * 2 - 1
rescaling_fn = lambda y: (y + 1) / 2 * (range_cns[1] - range_cns[0]) + range_cns[0]

mu_data = np.mean(CNs_ttest, axis=0)
std_data = np.std(CNs_ttest, axis=0)

scaled_cn_data = scaling_fn(cn_data)
train_data = torch.tensor(scaled_cn_data, dtype=torch.float32)

# data = torch.exp(data) # exponential
cn_stats = [np.min(scaled_cn_data), np.max(scaled_cn_data), np.mean(scaled_cn_data)]
print(cn_stats)

print ('training samples {}'.format(num_training_samples))
print ('targe data shape {}'.format(np.shape(CNs_ttest)))
print ('training data shape {}'.format(np.shape(cn_data)))

trainset = TensorDataset(train_data)
dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=args.flag_droplast, num_workers=int(args.num_workers))

num_training_samples = args.total_ads
# print(np.any(np.isnan(ADs)))
# print(np.any(np.isnan(CNs_ttest)))

# print(ADs)

# t-test
ttester = TTester(ADs[:args.total_ads])
targets, t_pvals, _ = ttester.get_bh_correction(CNs_ttest[:args.total_cns])
real_preds, r_pvals, _ = ttester.get_bh_correction(cn_data[:args.total_cns])
total_pos = np.sum(targets)
Helper.log(logf, 'Total-Rejects: {}'.format(total_pos))

k = 5
sorted_pval_idx = np.argsort(t_pvals)
significant_vertices = sorted_pval_idx[:k]
insignificant_vertices = list(reversed(sorted_pval_idx[-k:]))
print("Top significant vertices: {}, p_values: {}".format(significant_vertices, t_pvals[significant_vertices]))
print("Top insignificant vertices: {}, p_values: {}".format(insignificant_vertices, t_pvals[insignificant_vertices]))

## Add Gap here
if args.flag_mugap:
	print("Mu GAP")
	for i in range(len(targets)):
		if targets[i]:
			# adding to ADs to keep smoothness
			ADs[:, i] = ADs[:, i] + args.mu_gap
	ttester = TTester(ADs)
	# store
	Helper.log(logf, 'mu-gap: {}'.format(args.mu_gap))

####====== Modules =======####
def log_loss(epoch, step, total_step, D_cost, G_cost,  start_time):
	# convert
	# loss_d = D_cost.cpu().data.numpy()
	# loss_g = G_cost.cpu().data.numpy()
	loss_d = D_cost
	loss_g = G_cost
	# msg
	message = 'Epoch [{}/{}], Step [{}/{}], LossD: {:.4f}, LossG: {:.4f}, time: {:.4f}s'.format(epoch, args.num_epochs, step, total_step, loss_d, loss_g, time.time() - start_time)
	# log out
	Helper.log(logf, message)

def log_statistics(ads):
	# calculate Statistics
	mu_pred = torch.mean(ads, dim=0).to(device)
	std_pred = torch.std(ads, dim=0).to(device)
	# Root square error wrt the real statistics
	mu_rse = torch.norm(mu_pred - mu_data, p=2)
	std_rse = torch.norm(std_pred - std_data, p=2)
	# convert to cpu mode
	mu_rse = mu_rse.cpu().data.numpy()
	std_rse = std_rse.cpu().data.numpy()
	# log
	message = 'Statistics: mu-rse: {:.4f}, std-rse: {:.4f}'.format(mu_rse, std_rse)
	Helper.log(logf, message)

def log_ttest(labels, preds):
	# reject recall
	r = recall_score(labels, preds)
	p = precision_score(labels, preds)
	Helper.log(logf, 'Recall: {:.4f}, Precision: {:.4f}, #Rejects: {:d}'.format(r, p, np.sum(preds)))

def get_ttest(labels, preds, log=False):
	# reject recall
	r = recall_score(labels, preds)
	p = precision_score(labels, preds)
	if log:
		Helper.log(logf, 'Recall: {:.4f}, Precision: {:.4f}'.format(r, p))
	return p, r

def generate_sample(gen, nsamples):
	noise = torch.randn(nsamples, args.embed_size).to(device)
	# with torch.no_grad():
	# 	noisev = autograd.Variable(noise)
	return gen(noise)

def get_gradient_penalty(netD, real_data, fake_data):
	#
	alpha = torch.rand(real_data.shape[0], 1).to(device)
	alpha = alpha.expand(real_data.shape)
	#
	interpolates = alpha * real_data + ((1 - alpha) * fake_data)
	interpolates = interpolates.to(device)
	interpolates = autograd.Variable(interpolates, requires_grad=True)
	#
	disc_interpolates = netD(interpolates)
	#
	torch_ones = torch.ones(disc_interpolates.size()).to(device)
	grads = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch_ones, create_graph=True, retain_graph=True, only_inputs=True)
	grad = grads[0]
	# penalty
	grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
	return grad_penalty


"""
	# TODO: new paper
	same smoothness (not necessarily smooth)
	No normalized L
"""
def get_lapreg(f):
	# f: [N, 4225]
	# L: [4225, 4225]
	N = f.shape[0] # batch_size
	x1 = f.unsqueeze(1) # [N, 1, 4225]
	x2 = f.unsqueeze(2) # [N, 4225, 1]
	l = L.unsqueeze(0).expand(N, -1, -1)
	# [N, 1, 1]
	reg = torch.matmul(x1, torch.matmul(l, x2))
	reg = torch.sum(reg, dim=[1,2])
	reg = reg.mean()

	return reg

def get_statistics(fake_data):
	# mean Dim=0 or 1
	mu_sample = fake_data.mean(dim=0)
	std_sample = fake_data.std(dim=0)
	mu_rse = torch.norm(mu_sample - mu_data, p=1) #/ mu_data
	std_rse = torch.norm(std_sample - std_data, p=1) #/ std_data
	return mu_rse, std_rse

print("Stats on Real data")
log_ttest(targets, real_preds)
