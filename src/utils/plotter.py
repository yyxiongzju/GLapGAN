"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Loading data
"""

import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from sklearn.manifold import TSNE


SMALL_SIZE = 8.5
MEDIUM_SIZE = 14
BIGGER_SIZE = 20
plt.style.use('ggplot')
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE, titleweight='bold')     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE, labelweight='bold')    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# plt.rc('text', usetex=True)

class Plotter:

	@staticmethod
	def plot_precision():
		path = '../../result/'
		data1 = {} # nogap-precision
		data1['y_baseline'] = [0.2134, 0.2561, 0.3001, 0.3176, 0.3361, 0.3527, 0.3754, 0.3804, 0.3883, 0.3969, 0.4050, 0.4057, 0.4144, 0.4083, 0.4147, 0.4291, 0.4355, 0.4393]
		data1['y_lap'] = [0.2127, 0.3248, 0.3789, 0.3931, 0.3778, 0.3947, 0.4196, 0.4107, 0.4174, 0.4282, 0.4405, 0.4288, 0.4387, 0.4340, 0.4419, 0.4484, 0.4531, 0.4594]
		data1['title'] = 'Precision Varied with #-samples'
		data1['name'] = 'plot_precision_nogap'

		data2 = {}
		data2['y_lap'] = [0.2129, 0.3247, 0.3783, 0.3922, 0.3799, 0.3976, 0.4198, 0.4079, 0.4200, 0.4321, 0.4360, 0.4303, 0.4405, 0.4327, 0.4378, 0.4463, 0.4595, 0.4636]
		data2['y_baseline'] = [0.2134, 0.2561, 0.3001, 0.3176, 0.3363, 0.3529, 0.3754, 0.3804, 0.3884, 0.3969, 0.4050, 0.4059, 0.4144, 0.4083, 0.4147, 0.4291, 0.4355, 0.4393]
		data2['title'] = 'Precision Varied with #-samples with Gap'
		data2['name'] = 'plot_precision_gap'

		data = data2
		x = np.arange(100, 1000, 50)
		fig = plt.figure(figsize=(7, 6))
		plt.plot(x, data['y_lap'], label='GLapGAN', c='b')
		plt.plot(x, data['y_baseline'], label='Baseline', c='r')
		plt.title(data['title'])
		plt.xlabel("#-samples")
		plt.ylabel("Precision")
		plt.legend(loc='lower right')
		plt.savefig(path + '{}.png'.format(data['name']),
					bbox__hnches='tight')
		plt.close(fig)
		plt.show()

	@staticmethod
	def plot_murse():
		path = '../../result/'
		data1 = {} # nogap-precision
		data1['y_baseline'] = [22581.7285, 7076.2593, 2865.0449, 2651.5859, 19100.9160, 22305.7461, 5009.4980, 18213.1504, 552.3261, 24253.8047, 23571.0508, 14978.4141, 8663.2012, 59667.5312, 86321.2734, 185651.5156, 6351.8491, 4099.7300]

		data1['y_lap'] = [268.5231, 8.0874, 8.0858, 8.0324, 7.7938, 7.8150, 7.7954, 7.7535, 7.7365, 7.7236, 7.6339, 7.6100, 7.4337, 7.4708, 7.4691, 7.4059, 7.3455, 7.4087]
		data1['title'] = 'Mean-RSE Varied with #-samples'
		data1['name'] = 'plot_mu_nogap'

		data2 = {}
		data2['y_baseline'] = [23535.80, 6932.221, 2143.240, 6332.128, 17772.12, 8083.403, 11431.30, 37872.10, 745.3015, 15031.16, 9018.014, 1664.217, 63276.74, 151258.6, 4826.481, 5878.041, 484260.5, 1246.513]
		data2['y_lap'] = [228.5559, 8.0789, 8.1865, 8.0680, 7.7946, 7.8001, 7.7838, 7.7502, 7.7333, 7.5946, 7.6719, 7.5811, 7.1163, 7.5014, 7.6529, 7.4961, 7.3265, 7.3687]
		data2['title'] = 'Mean-RSE Varied with #-samples with Gap'
		data2['name'] = 'plot_mu_gap'

		data = data1
		x = np.arange(100, 1000, 50)
		fig = plt.figure(figsize=(7, 6))
		plt.plot(x, data['y_lap'], label='GLapGAN', c='b')
		plt.plot(x, data['y_baseline'], label='Baseline', c='r')
		plt.title(data['title'])
		plt.xlabel("#-samples")
		plt.ylabel("Mean-RSE")
		plt.legend(loc='lower right')
		plt.savefig(path + '{}.png'.format(data['name']),
					bbox__hnches='tight')
		plt.close(fig)
		plt.show()

	@staticmethod
	def save_images(samples, im_size, path, idx, n_fig_unit=2):
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(n_fig_unit, n_fig_unit)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(im_size, im_size), cmap='Greys_r')

		plt.savefig(path + '/{}.png'.format(str(idx).zfill(3)),
					bbox__hnches='tight')
		plt.close(fig)
		return fig

	@staticmethod
	def plot_sample(samples, real_data):
		x = np.mean(samples, axis=0)
		y = np.mean(real_data, axis=0)
		plt.scatter(range(100), x[:100])
		plt.scatter(range(100), y[:100])
		plt.show()

		sample_mu = np.mean(samples, axis=0)
		sample_std = np.std(samples, axis=0)
		data_mu = np.mean(real_data, axis=0)
		data_std = np.std(real_data, axis=0)
		print("Mu: ", np.linalg.norm(sample_mu - data_mu))
		print("std: ", np.linalg.norm(sample_std - data_std))


	@staticmethod
	def plot_hist_1(data, deg_vec, path, idx):
		x = data / deg_vec
		fig = plt.figure(figsize=(4, 4))
		plt.scatter(np.arange(len(x)), x)
		plt.savefig(path + '/{}.png'.format(str(idx).zfill(3)),
					bbox__hnches='tight')
		plt.close(fig)

	@staticmethod
	def plot_hist_2(data, deg_vec):
		fig = plt.gcf()
		fig.show()
		fig.canvas.draw()
		plt.title("Gaussian Histogram")
		plt.xlabel("Value")
		plt.ylabel("Frequency")
		for node in range(len(deg_vec)):
			try:
				x = data[:, node] / deg_vec[node]
				mu = np.mean(x)
				sig = np.std(x)
				print('Node {:d}: mean:{:.3f}, std: {:.3f}'.format(node, mu, sig))
				plt.hist(x, 20) # Hard-code
				fig.canvas.draw()
				input('Press to continue ...')
			except:
				# from IPython import embed; embed() #os._exit(1)
				print('Exception')
				break

	@staticmethod
	def plot_fig(lmse):
		x = np.arange(len(lmse))
		plt.figure()
		plt.plot(x, results[0], c='r')
		plt.plot(x, results[1], c='b')
		plt.plot(x, results[-1], c='g')

	@staticmethod
	def plot_tnse(fname):
		data = np.load(fname)
		tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
		result = tsne.fit_transform(data)
		vis_x = result[:, 0]
		vis_y = result[:, 1]
		plt.scatter(vis_x, vis_y, cmap=plt.cm.get_cmap("jet", 10))
		plt.title('Laplacian')
		plt.show()

	@staticmethod
	def plot_dist(dists, n_pixels, name):
		# Plot the distributions
		# from IPython import embed; embed()
		means = dists[:, 0]
		sds = dists[:, 1]
		hpd_025 = dists[:, 2]
		hpd_975 = dists[:, 3]

		fig = plt.figure(figsize=(20, 7))
		plt.errorbar(range(n_pixels), means, yerr=sds, fmt='-o', linestyle="None")
		plt.scatter(range(n_pixels), hpd_025, c='b')
		plt.scatter(range(n_pixels), hpd_975, c='b')
		fig.tight_layout()
		fig.savefig(name, bbox_inches='tight')
		plt.xlabel('Vertex')
		plt.title('Difference of means distribution')
		plt.show()

	@staticmethod
	def plot_signals(off_data, gan_name, cn):
		root_path = '../result'
		fig = plt.figure(figsize=(10, 7))
		name_data = offdata2name(off_data)
		if cn:
			cn_name = 'cn'
		else:
			cn_name = 'ad'
		plot_path = "{}/{}/plt_{}_{}.png".format(root_path, gan_name, name_data, cn_name)
		data_path = '../data/{}/data_{}_4k.mat'.format(name_data, name_data)
		_, real_signals = load_data(data_path, is_control=cn)
		for off_model in range(1, 4):
			name_model = offmodel2name(off_model)
			if off_model == 3:
				signals = real_signals
				if off_data == 3: # Simuln
					signals = signals[:1000, :]
			else:
				sample_path = "{}/{}/{}/{}".format(root_path, gan_name, name_data, name_model)
				signal_path = os.path.join(sample_path, "{}/samples/samples_1000.npy".format(cn_name))
				signals = np.load(signal_path)[:real_signals.shape[0], :]
			means = np.mean(signals, axis = 0)
			plt.scatter(range(len(means)), means, label=name_model)
		plt.legend()
		plt.xlabel('Node')
		plt.ylabel("Value")
		plt.title('{} signals on {} - {}'.format(cn_name, name_data, gan_name))
		plt.show()
		fig.tight_layout()
		fig.savefig(plot_path, bbox_inches='tight')

	@staticmethod
	def plot_eb(off_data, off_gan, alp, epsilon):
		fig = plt.figure(figsize=(10, 7))
		name_data = offdata2name(off_data)
		name_gan = offgan2name(off_gan)
		root_folder = "../result/eb-same/{}".format(name_gan)
		plot_path = "{}/plt_{}_{}_{}_{}.png".format(root_folder, name_data, name_gan, alp, epsilon)
		for off_model in range(1, 4):
			name_model = offmodel2name(off_model)
			fname = "eval_{}_{}_{}".format(name_data, name_gan, name_model)
			if off_model ==1:
				file_path = "{}/{}_{}_{}.npy".format(root_folder, fname, alp, epsilon)
			elif off_model == 2:
				file_path = "{}/{}_{}.npy".format(root_folder, fname, epsilon)
			else:
				sample_folder = '../result/eb-same/real'
				file_path = "{}/eval_{}_{}.npy".format(sample_folder, name_data,  epsilon)
			arr = np.load(file_path)
			# plt.plot(range(len(arr)), sorted(arr), label=name_model)
			plt.scatter(range(len(arr)), arr, label=name_model)
		plt.legend()
		plt.xlabel('Node')
		plt.ylabel("e-value")
		plt.title('{} data - e-value on {} - {} with epsilon = {}'.format(name_data, name_gan, alp, epsilon))
		plt.show()

	@staticmethod
	def plot_ttest_alp(off_data, off_gan, alp):
		name_data = offdata2name(off_data)
		name_gan = offgan2name(off_gan)
		root_path = '../result'
		output_path = "{}/ttest/{}".format(root_path, name_gan)
		plot_path = "{}/plt_{}_{}_{}.png".format(output_path, name_data, name_gan, alp)

		fig = plt.figure(figsize=(10, 7))
		if off_data == 4:
			model_list = [2, 3]
		else:
			model_list = [1, 2, 3]

		colors = ['y', 'b', 'r', 'g']
		for off_model in model_list:
			name_model = offmodel2name(off_model)
			if off_model == 1:
				file_path = "{}/{}_{}_{}_{}.npy".format(output_path, name_data, name_gan, name_model, alp)
			else:
				file_path = "{}/{}_{}_{}_0.01.npy".format(output_path, name_data, name_gan, name_model)
			arr = np.load(file_path)
			spio.savemat('/home/tuandinh/Documents/Project/glapGan/data/mesh_all/pvalue_{}_{}.mat'.format(name_model, name_data), {'pvalue_{}'.format(name_model):arr})
			return
			# from IPython import embed; embed()
			# idx = arr < 0.08
			# arr = arr[idx]
			plt.plot(range(len(arr)), sorted(arr), label=name_model,  c=colors[off_model])
			# plt.scatter(range(len(arr)), arr, label=name_model)

		plt.legend()
		plt.xlabel('Node')
		plt.ylabel("p-value")
		plt.title('{} data - BH correction with {} - alpha {}'.format(name_data, name_gan, alp))
		plt.show()
		fig.tight_layout()
		fig.savefig(plot_path, bbox_inches='tight')

	@staticmethod
	def plot_ttest_all(off_data, off_gan):
		name_data = offdata2name(off_data)
		name_gan = offgan2name(off_gan)
		root_path = '../result'
		output_path = "{}/ttest/{}".format(root_path, name_gan)
		plot_path = "{}/plt_{}_{}_all.png".format(output_path, name_data, name_gan)

		fig = plt.figure(figsize=(7, 4))
		if off_data == 4:
			model_list = [2, 3]
		else:
			model_list = [1, 2, 3]

		colors = ['y', 'b', 'r', 'g']
		for off_model in [2, 3]:
			name_model = offmodel2name(off_model)
			file_path = "{}/{}_{}_{}_0.01.npy".format(output_path, name_data, name_gan, name_model)
			arr = np.load(file_path)
			# idx = arr < 0.08
			# arr = arr[idx]
			plt.plot(range(len(arr)), sorted(arr), label=name_model, c=colors[off_model])

		linestyles = ['-', '--', '-.', ':']
		i = 0
		for alp in [0.03, 0.07, 0.11, 0.15]:
			name_model = offmodel2name(1)  # lapgan
			file_path = "{}/{}_{}_{}_{}.npy".format(output_path, name_data, name_gan, name_model, alp)
			arr = np.load(file_path)
			plt.plot(range(len(arr)), sorted(arr), label=name_model + ' ' r'$\alpha=$'+str(alp), linestyle=linestyles[i], c='b')
			i = i + 1

		plt.legend()
		plt.xlabel('Node')
		plt.ylabel("p-value")
		plt.title('{} data - BH correction with {}'.format(name_data, name_gan))
		plt.show()
		fig.tight_layout()
		fig.savefig(plot_path, bbox_inches='tight')

	@staticmethod
	def plot_ttest_zoom(off_data, off_gan):
		name_data = offdata2name(off_data)
		name_gan = offgan2name(off_gan)
		root_path = '../result'
		output_path = "{}/ttest/{}".format(root_path, name_gan)
		plot_path = "{}/plt_{}_{}_zoom.png".format(output_path, name_data, name_gan)

		# fig = plt.figure(figsize=(7, 4))
		fig, ax = plt.subplots() #
		x = range(4225)
		linestyles = ['-', '--', '-.', ':']
		colors = ['y', 'b', 'r', 'g']
		for off_model in [2, 3]:
			name_model = offmodel2name(off_model)
			file_path = "{}/{}_{}_{}_0.01.npy".format(output_path, name_data, name_gan, name_model)
			arr = np.load(file_path)
			ax.plot(x, sorted(arr), label=name_model, c=colors[off_model])

		i = 0
		for alp in [0.03, 0.07, 0.11, 0.15]:
			name_model = offmodel2name(1)  # lapgan
			file_path = "{}/{}_{}_{}_{}.npy".format(output_path, name_data, name_gan, name_model, alp)
			arr = np.load(file_path)
			ax.plot(x, sorted(arr), label=r'$\lambda_L=$'+str(alp), linestyle=linestyles[i], c='b')
			i = i + 1

		plt.legend(loc='lower right')
		plt.xlabel('Node (arbitrary order)')
		plt.ylabel("p-value")
		plt.title('Sorted p values after Benjamini-Hochberg correction')

		axins = zoomed_inset_axes(ax, 3, loc='upper left', bbox_to_anchor=(0.16, 0.9),bbox_transform=ax.figure.transFigure) # zoom-factor: 2.5,
		# axins = inset_axes(ax, 1,1 , loc=2,bbox_to_anchor=(0.2, 0.55))
		for off_model in [2, 3]:
			name_model = offmodel2name(off_model)
			file_path = "{}/{}_{}_{}_0.01.npy".format(output_path, name_data, name_gan, name_model)
			arr = np.load(file_path)
			axins.plot(x, sorted(arr), label=name_model, c=colors[off_model])

		i = 0
		for alp in [0.03, 0.07, 0.11, 0.15]:
			name_model = offmodel2name(1)  # lapgan
			file_path = "{}/{}_{}_{}_{}.npy".format(output_path, name_data, name_gan, name_model, alp)
			arr = np.load(file_path)
			axins.plot(x, sorted(arr), label=r'$\alpha=$'+str(alp), linestyle=linestyles[i], c='b')
			i = i + 1

		x1, x2, y1, y2 = 1100, 2100, 0, 0.08 # specify the limits
		axins.set_xlim(x1, x2) # apply the x-limits
		axins.set_ylim(y1, y2) # apply the y-limits
		axins.set_facecolor((1, 0.75, 0.75))
		mark_inset(ax, axins, loc1=1, loc2=3, linewidth=1, ec="0.5")
		# plt.yticks(visible=False)
		plt.xticks(visible=False)
		plt.grid(False)
		fig.tight_layout()
		fig.savefig(plot_path)
		plt.show()

	@staticmethod
	def get_fdr(r_pvalues, l_pvalues):
		n = 20
		t = np.linspace(0.01, 0.1, num=n)
		lines = np.zeros((n, 1))
		for k in range(n):
			threshold = t[k]
			l_pred = np.asarray(l_pvalues < threshold, dtype=int)
			r_pred = np.asarray(r_pvalues < threshold, dtype=int)
			l_v = 0
			for i in range(len(r_pred)):
				if r_pred[i] == 1:
					l_v += l_pred[i]
			# l_v = np.sum(abs(l_pred - r_pred))
			# b_v = np.sum(abs(b_pred - r_pred))
			lines[k] = l_v / np.sum(r_pred)
			# from IPython import embed; embed()
		return lines

	@staticmethod
	def plot_fdr(off_data, off_gan):
		name_data = offdata2name(off_data)
		name_gan = offgan2name(off_gan)
		root_path = '../result'
		output_path = "{}/ttest/{}".format(root_path, name_gan)
		plot_path = "{}/recall_{}_{}_all.png".format(output_path, name_data, name_gan)

		fig = plt.figure(figsize=(10, 7))
		b_pvalues = np.load("{}/{}_{}_{}_0.01.npy".format(output_path, name_data, name_gan, offmodel2name(2)))
		r_pvalues = np.load("{}/{}_{}_{}_0.01.npy".format(output_path, name_data, name_gan, offmodel2name(3)))

		b_line = get_fdr(r_pvalues, b_pvalues)
		plt.plot(t, b_line, label='WGAN (baseline)', c='r')
		linestyles = ['-', '--', '-.', ':']
		colors = ['y', 'b', 'r', 'g']
		i = 0
		for alp in [0.05, 0.1, 0.15]:
			name_model = offmodel2name(1)  # lapgan
			file_path = "{}/{}_{}_{}_{}.npy".format(output_path, name_data, name_gan, name_model, alp)
			l_pvalues = np.load(file_path)
			l_line = get_fdr(r_pvalues, l_pvalues)
			plt.plot(t, l_line, label=r'$\lambda_L=$'+str(alp), linestyle=linestyles[i], c='b')
			i = i + 1
			# from IPython import embed; embed()

		plt.legend()
		plt.xlabel('p-value threshold')
		plt.ylabel("Sensitivity")
		plt.grid(b=True)
		plt.title('Sensitivity of t-test with generated data');
		plt.show()
		fig.tight_layout()
		fig.savefig(plot_path, bbox_inches='tight')

if __name__ ==  '__main__':
	Plotter.plot_murse()
