"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Training 2 signals together
"""

from scipy.stats import ttest_ind
from statsmodels.sandbox.stats.multicomp import multipletests


class TTester:
	"""
		@input: cns = [N, D]
	"""

	def __init__(self, ads, coeff=0.05):
		self.ads = ads
		self.coeff = coeff

	def set_coeff(self, coeff):
		self.coeff = coeff

	def get_t_stats(self, ad, cn):
		# For each voxel
		twosample_results = ttest_ind(ad, cn)
		return twosample_results[1]

	def get_pvalues(self, cns):
		num_voxel = cns.shape[1]
		pvalues = [self.get_t_stats(cns[:, i], self.ads[:, i])
				   for i in range(num_voxel)]
		return pvalues

	def get_bh_correction(self, cns):
		pvalues = self.get_pvalues(cns)
		rejects, corrected_pvalues, _, _ = multipletests(
			pvalues, alpha=self.coeff, method='fdr_bh')

		return rejects, corrected_pvalues, pvalues
