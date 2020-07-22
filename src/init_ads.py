"""
    @author Yunyang/Eric/Tuan
    @date 06/24/2020
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

from configs_ads import args
from utils.helper import Helper, AverageMeter
from utils.provider import Provider
from utils.ttester import TTester
from models.wgan import *

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
device = args.gpu
ADs, CNs, A, L = Provider.load_data(args.dataset_path, dataset_name)

if args.mode_dataset == 1:
    A = Helper.sparse_mx_to_torch_sparse_tensor(A)
    L = L.todense()

L = torch.tensor(L, dtype=torch.float32).to(device)
A = torch.tensor(A, dtype=torch.float32).to(device)

nsamples = args.num_ads
num_training_samples = args.num_ads
CNs_ttest = ADs[:nsamples]
cn_data = ADs[:nsamples]

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

cns_netG = Generator(args.embed_size, args.signal_size).to(device)
cns_netG.apply(Helper.weights_init)
print("#-cns: {}".format(args.num_cns))
dir_cns_path = ''
if args.num_cns <= 293:
    dir_cns_path = './results/lwgan-adni-{}-reg'.format(args.num_cns)
else:
    dir_cns_path = './results/lwgan-adni-emci-{}-reg'.format(args.num_cns)

cns_model_path = os.path.join(dir_cns_path, 'snapshots-{}'.format(args.num_cns))
cns_netG_path = os.path.join(cns_model_path, 'netG.pth')
cns_netG.load_state_dict(torch.load(cns_netG_path))

def generate_sample(gen, nsamples):
    noise = torch.randn(nsamples, args.embed_size).to(device)
    return gen(noise)

num_training_samples = args.num_cns
cns_fake = generate_sample(cns_netG, args.num_cns)
rescaled_cns_fake = rescaling_fn(cns_fake)

# t-test
ttester = TTester(CNs[:args.num_cns])
targets, t_pvals, _ = ttester.get_bh_correction(ADs[:args.total_ads])
real_preds, r_pvals, _ = ttester.get_bh_correction(ADs[:args.total_ads])
total_pos = np.sum(targets)
Helper.log(logf, 'Total-Rejects: {}'.format(total_pos))

fake_ttester = TTester(rescaled_cns_fake.cpu().data.numpy())
fake_targets, fake_t_pvals, _ = fake_ttester.get_bh_correction(ADs[:args.total_ads])
fake_real_preds, fake_r_pvals, _ = fake_ttester.get_bh_correction(ADs[:args.total_ads])
fake_total_pos = np.sum(fake_targets)
Helper.log(logf, 'GAN-generated Total-Rejects: {}'.format(fake_total_pos))

out_fake_real_data_path = os.path.join(args.dataset_path, '{}/data_adni_fake_real.npy'.format(dataset_name))
mat_out_fake_real_data_path = os.path.join(args.dataset_path, '{}/data_adni_fake_real.mat'.format(dataset_name))

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
            CNs[:, i] = CNs[:, i] + args.mu_gap
    ttester = TTester(CNs[:args.num_cns])
    # store
    Helper.log(logf, 'mu-gap: {}'.format(args.mu_gap))

####====== Modules =======####
def log_loss(epoch, step, total_step, D_cost, G_cost,  start_time):
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
