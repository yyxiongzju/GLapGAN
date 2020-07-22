"""
	@author Yunyang/Eric/Tuan
	@date 06/24/2020
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode_dataset', type=int, default=0)
parser.add_argument('--mode_datatype', type=int, default=0)
parser.add_argument('--mode_model', type=int, default=0)
parser.add_argument('--num_ads', type=int, default=183)
parser.add_argument('--num_cns', type=int, default=293)
parser.add_argument('--total_ads', type=int, default=183)
parser.add_argument('--total_cns', type=int, default=293)
parser.add_argument('--train_fraction', type=float, default=1.)
parser.add_argument('--mu_gap', type=float, default=0.5)
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

parser.add_argument('--result_path', type=str, default='results', help='output path')
parser.add_argument('--dataset_path', type=str, default='../data', help='path for data')

# Training
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--critic_steps', type=int, default=3)
parser.add_argument('--lr_g', type=float, default=0.1)
parser.add_argument('--lr_d', type=float, default=0.1)
parser.add_argument('--lam', type=float, default=0.1)
parser.add_argument('--alpha', type=float, default=0.1)


parser.add_argument('--log_epochs', type=int, default=1, help='step size for prining log info')
parser.add_argument('--save_step', type=int, default=10, help='step size for saving trained models')

parser.add_argument('--isTest', default=False, action='store_true', help='Test')
parser.add_argument('--flag_retrain', default=False, action='store_true', help='Re train')
parser.add_argument('--flag_reg', default=False, action='store_true', help='Regularizer')
parser.add_argument('--flag_ttest', default=False, action='store_true', help='Run ttest')
parser.add_argument('--flag_plot', default=False, action='store_true', help='Plot')
parser.add_argument('--flag_moment', default=False, action='store_true', help='Moments matching')
parser.add_argument('--flag_mugap', default=False, action='store_true', help='Moments matching')
parser.add_argument('--flag_droplast', default=False, action='store_true', help='Drop Last')

# Model parameters
parser.add_argument('--signal_size', type=int, default=4059)
parser.add_argument('--embed_size', type=int, default=100)

parser.add_argument('--num_workers', type=int, default=0)

args = parser.parse_args()
# print(args)
