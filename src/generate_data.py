from utils.provider import Provider

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode_dataset', type=int, default=0)
parser.add_argument('--dataset_path', type=str, default='../data', help='path to data folder')
parser.add_argument('--extra_samples', type=int, default=2000)
args = parser.parse_args()

## Data path
DATASETS = ['adni', 'adrc', 'simuln', 'synthetic']
dataset_name = DATASETS[args.mode_dataset]

Provider.generate_data(args.dataset_path, dataset_name, extra=args.extra_samples, sparse=False)
print('Generate data for {}'.format(dataset_name))
