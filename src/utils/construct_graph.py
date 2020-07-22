"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Generate Simulated Data For Testing
"""
import sys
import numpy as np
from ttester import TTester
from sklearn.metrics import recall_score, precision_score
from matplotlib import pyplot as plt

class GraphConstructor():
	"""
			Graph Constructor class
	"""

	def __init__(self, dim=4225, smoothness_level=0.1, mu_gap=0.5, reject_fraction=0.1):
		self.dim = dim  # dimension of signals
		self.mu_gap = mu_gap  # the gap between 2 means
		self.smoothness_level = smoothness_level
		self.num_rejects = int(self.dim * reject_fraction)
		self.A, self.L = self._construct()

	def _construct(self):
		# sparse graph
		A = self.create_random_connected_graph(p=0.3)
		# diagonal matrix
		D = np.sum(A, axis=0)**(-0.5)
		D = np.diag(D)
		# Lapalacian matrix
		L = np.eye(self.dim) - np.matmul(D, np.matmul(A, D))
		return A, L

	def get_properties(self):
		return self.A, self.L

	def get_smoothness(self, x):
		"""
				* L: [d, d] normalized Lapalacian matrix
				* x: [d, 1] signal
				return smoothness of x wrt L
		"""
		s = self.L.dot(x)
		s = s.dot(x)
		return s

	def create_random_connected_graph(self, p):
		"""
				* p: probability of adding an edge
		"""
		A = np.zeros(shape=(self.dim, self.dim))
		# connected
		for i in range(self.dim-1):
			A[i, i + 1] = 1
			A[i + 1, i] = 1

		# randomly add edges with probability p
		for i in range(self.dim-1):
			for j in range(i + 1, self.dim):
				t = np.random.rand()
				if t < p:
					A[i, j] = 1
					A[j, i] = 1
		return A

	def generate_simulated_data(self, num_samples=1000, max_steps=100):
		"""
				Generate simulated multivariate signals based on theory-given assumptions
				* dim: dimension of signals
				* num_rejects: number of detected vertices
				* mu_cn, mu_ad: [N, dim]
				* sig: [dim, dim]
				Using a same covariance matrix for both signals as the theory
				The test is very sensitive to the regression, hence, we should not normalize?
		"""
		# means of cn: this is very important for the test
		mu_cn = np.random.rand(self.dim)
		# smooth the signal until the smoothness is satisfied
		step = 0
		# np.sum(self.A, axis=0)
		frob_norm = np.max(np.sum(self.A, axis=0))
		while self.get_smoothness(mu_cn) > self.smoothness_level and step < max_steps:
			mu_cn = self.A.dot(mu_cn)/frob_norm
			step += 1
		# random locations
		print('Step: ', step)
		mu_ad = np.copy(mu_cn)

		# locs = np.random.randint(low=0, high=self.dim-1, size=self.num_rejects)
		pivot = np.random.randint(low=1, high=self.dim-self.num_rejects-1)
		locs = np.arange(self.num_rejects) + pivot
		# Ads data mu
		for i in range(self.num_rejects):
			loc = locs[i]
			mu_ad[loc] += self.mu_gap
		# covariance
		cov = np.random.rand(self.dim, self.dim)
		# sig = np.dot(cov, cov.transpose()) / self.dim
		sig = np.identity(self.dim)
		# generate cns samples
		cns = np.random.multivariate_normal(mu_cn, sig, num_samples)
		ads = np.random.multivariate_normal(mu_ad, sig, num_samples)
		return cns, ads, mu_cn, mu_ad


if __name__ == "__main__":
	# input
	if len(sys.argv) > 1:
		dim = int(sys.argv[1])
		num_samples = int(sys.argv[2])
	else:
		dim = 1000
		num_samples = 30000

	flag_load_data = True
	data_path = '../../data/synthetic/data.npy'
	if flag_load_data:
		print('Load data')
		data = np.load(data_path, allow_pickle=True)
		ads = data.item()['ad']
		cns = data.item()['cn']
	else:
		print('Generate data')
		# instance
		gc = GraphConstructor(dim=dim, smoothness_level=0.1, mu_gap=0.05, reject_fraction=0.1)
		A, L = gc.get_properties()
		# ---- #
		cns, ads, mu_cn, mu_ad = gc.generate_simulated_data(num_samples=num_samples)
		data = {'ad': ads, 'cn': cns, 'L':L, 'A':A}
		np.save(data_path, data)

	ttester = TTester(ads)
	targets, _, _ = ttester.get_bh_correction(cns)
	print('#rejects: ', np.sum(targets))
	# fig = plt.figure()
	# plt.plot(range(dim), np.mean(ads, axis=0), label='ad', c='b')
	# plt.plot(range(dim), np.mean(cns, axis=0), label='cn', c='g')
	# plt.legend(loc='lower right')
	# plt.show()
	pscores = np.zeros(10)
	rscores = np.zeros(10)
	xrange = np.arange(1, 11) * 0.1
	nIters = 50
	for i in range(10):
		num_cns = int(xrange[i] * num_samples)
		p = 0
		r = 0
		for k in range(nIters):
			locs = np.random.choice(np.arange(num_samples), size=num_cns, replace=False)
			X_cn = cns[locs,:]
			preds, _, _ = ttester.get_bh_correction(X_cn)
			p += precision_score(targets, preds)
			r += recall_score(targets, preds)
		print(i, ' #scores: ', p, r)
		pscores[i] = p / nIters
		rscores[i] = r / nIters


	fig_path = '../../results/'
	fig = plt.figure(figsize=(7, 6))
	plt.plot(xrange, pscores, label='precision', c='b')
	plt.plot(xrange, rscores, label='recall', c='g')
	plt.title('Precision over #-samples')
	plt.xlabel("#-samples")
	plt.ylabel("T-Test")
	plt.legend(loc='lower right')
	# plt.show()
	plt.savefig(fig_path + 'real_data.png', bbox__hnches='tight')
	plt.close(fig)
	from IPython import embed; embed()

	# results from current dataset
	'''
		0  #rejects:  48.60079365079365 5.85
		1  #rejects:  49.620795499188745 28.939999999999998
		2  #rejects:  49.7359650796594 40.88
		3  #rejects:  49.787548677975614 46.19999999999998
		4  #rejects:  49.63939525427713 48.63999999999999
		5  #rejects:  49.760479104333 49.49999999999999
		6  #rejects:  49.78275109178534 49.94
		7  #rejects:  49.83226557949912 49.99
		8  #rejects:  49.930693069306926 50.0
		9  #rejects:  50.0 50.0
	'''
