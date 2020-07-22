"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Training 2 signals together
"""

from init import *
from models.wgan import Generator, Discriminator
from matplotlib import pyplot as plt
plt.ion()

#### ==================Model======================
netG = Generator(args.embed_size, args.signal_size).to(device)
netD = Discriminator(args.signal_size).to(device)
netD.apply(Helper.weights_init)
netG.apply(Helper.weights_init)

optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(0.5, 0.9))

# optimizerD = optim.RMSprop(netD.parameters(), lr=args.lr, weight_decay=1e-4)
# optimizerG = optim.RMSprop(netG.parameters(), lr=args.lr, weight_decay=1e-4)

#### ==================Training======================
def trainD(batch_data, penalty=False):
	# data
	real_data = batch_data[0].to(device)
	with torch.no_grad():
		real_data_zero = autograd.Variable(real_data)
	# train real
	realD = netD(real_data_zero)

	# train with fake
	noise = torch.randn(real_data_zero.shape[0], args.embed_size).to(device)
	with torch.no_grad():
		noise_zero = autograd.Variable(noise)  # totally freeze netG
	fake_data = autograd.Variable(netG(noise_zero).data)
	#### This will prevent gradient on params of G
	fakeD = netD(fake_data)
	if penalty:
		# train with gradient penalty
		# gradient_penalty = get_gradient_penalty(netD, real_data_zero.data, fake_data.data)
		grad_penalty = get_gradient_penalty(netD, real_data_zero, fake_data)
	else:
		grad_penalty = None

	return realD, fakeD, grad_penalty

def trainG():
	noise = torch.randn(args.batch_size, args.embed_size).to(device)
	with torch.no_grad():
		noise_zero = autograd.Variable(noise)
	fake_data = netG(noise_zero)
	fakeD = netD(fake_data)
	return fakeD, fake_data

start_time = time.time()
if os.path.isfile(netG_path) and args.flag_retrain:
	print('Load existing models')
	netG.load_state_dict(torch.load(netG_path))
	netD.load_state_dict(torch.load(netD_path))

vis_fig, vis_axes = plt.subplots(2, k)
hist_bins = np.linspace(0, 5, 50)
plt.show()
for epoch in range(args.num_epochs):
	step = 0
	data_iter = iter(dataloader)
	total_step = len(data_iter)
	while step < total_step:
		vis_fig.canvas.flush_events()
		# (1) Update D network
		Helper.activate(netD)
		stepD = 0
		while stepD < args.critic_steps and step < total_step - 1:
			# for p in netD.parameters():
			# 	p.data.clamp_(-0.1, 0.1)

			optimizerD.zero_grad()
			realD, fakeD, grad_penalty = trainD(next(data_iter), penalty=True)
			realD = realD.mean()
			fakeD = fakeD.mean()
			# cost
			costD = fakeD - realD + args.lam * grad_penalty
			costD.backward()
			optimizerD.step()
			stepD += 1
			step += 1
		# (2) Update G network
		Helper.deactivate(netD)
		optimizerG.zero_grad()
		fakeD, fake_data = trainG()
		fakeD = fakeD.mean()
		costG = -fakeD

		# laplacian regularizer
		if args.flag_reg:
			lapreg = args.alpha * get_lapreg(fake_data)
			costG += lapreg

		# backward
		costG.backward()
		optimizerG.step()

		# Print log info
		# if 0 == step % args.log_step:
		log_loss(epoch, step, total_step, costD, costG, start_time)
		start_time = time.time()
		step += 1
	
	if (epoch + 1) % args.log_epochs == 0:

		cns_np = rescaling_fn(generate_sample(netG, num_training_samples).detach().cpu().numpy())
		
		mu_pred = np.mean(cns_np, axis=0)
		std_pred = np.std(cns_np, axis=0)
		mu_me = np.mean(np.abs(mu_pred - mu_data))
		std_me = np.mean(np.abs(std_pred - std_data))
		print('Moments: mean-me {:.4f} std-me: {:.4f}'.format(mu_me, std_me))

		# run ttest
		if args.flag_ttest:
			# generated samples
			ps = 0
			rs = 0
			preds, pval, _  = ttester.get_bh_correction(cns_np)
			print("#Rejects: {:d}".format(np.sum(preds)))
			# compare with ground-truth
			p, r = get_ttest(targets, preds)
			print(targets)
			print(preds)
			ps += p
			rs += r
			print('Recall: {:.4f}, Precision: {:.4f}'.format(rs, ps))

		# Visualizations
		for i in range(k):
			v_idx = significant_vertices[i]
			ax = vis_axes[0, i]
			ax.clear()
			ax.hist(ADs[:, v_idx], hist_bins, density=True, alpha=0.4, label="AD")
			ax.hist(cn_data[:, v_idx], hist_bins, density=True, alpha=0.4, label="CN")
			ax.hist(cns_np[:, v_idx], hist_bins, density=True, alpha=0.4, label="Fake CN")
			ax.legend()
			title_string = "significant vertex {:d}\n μ offset: {:.2f}, σ offset: {:.2f}".format(v_idx, 
																								mu_pred[v_idx] - mu_data[v_idx], 
																								std_pred[v_idx] - std_data[v_idx])
			if args.flag_ttest:
				title_string += "\nAdjusted pval: {:.2E}, {}".format(pval[v_idx], "Rejected" if preds[v_idx] else "Not rejected")
			ax.set_title(title_string)
		
		for i in range(k):
			v_idx = insignificant_vertices[i]
			ax = vis_axes[1, i]
			ax.clear()
			ax.hist(ADs[:, v_idx], hist_bins, density=True, alpha=0.4, label="AD")
			ax.hist(cn_data[:, v_idx], hist_bins, density=True, alpha=0.4, label="CN")
			ax.hist(cns_np[:, v_idx], hist_bins, density=True, alpha=0.4, label="Fake CN")
			ax.legend()
			title_string = "insignificant vertex {:d}\n μ offset: {:.2f}, σ offset: {:.2f}".format(v_idx, 
																								mu_pred[v_idx] - mu_data[v_idx], 
																								std_pred[v_idx] - std_data[v_idx])
			if args.flag_ttest:
				title_string += "\nAdjusted pval: {:.2E}, {}".format(pval[v_idx], "Rejected" if preds[v_idx] else "Not rejected")
			ax.set_title(title_string)
		
		vis_fig.canvas.draw()
		vis_fig.canvas.flush_events()

		
		print('save models at epoch')
		torch.save(netG.state_dict(), netG_path)
		torch.save(netD.state_dict(), netD_path)

		# from IPython import embed; embed()
