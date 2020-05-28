import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn.functional as F

torch.backends.cudnn.bencmark = True
import numpy as np
import random
import sys, cv2, dill


class L2Norm(nn.Module):
	def __init__(self):
		super(L2Norm, self).__init__()
		self.eps = 1e-10

	def forward(self, x):
		norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
		x = x / norm.unsqueeze(-1).expand_as(x)
		return x


class HardNet(nn.Module):
	def __init__(self, in_c=1, size=32, dropout=0.3, loss='triplet_margin', coordconv=False, affine=False):
		super(HardNet, self).__init__()
		self.size = size
		self.loss = loss
		self.coordconv = coordconv
		if self.coordconv:
			in_c += 2
		self.features = nn.Sequential(
			nn.Conv2d(in_c, 32, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(32, affine=affine),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(32, affine=affine),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(64, affine=affine),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(64, affine=affine),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(128, affine=affine),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(128, affine=affine),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Conv2d(128, 128, kernel_size=8, bias=False),
			nn.BatchNorm2d(128, affine=affine)
		)
		self.features.apply(weights_init)

	def forward(self, input):
		x_features = self.features(self.input_norm(input))
		if self.size > 32:
			x_features = nn.AdaptiveAvgPool2d((1, 1))(x_features)
		x = x_features.view(x_features.size(0), -1)
		if self.training and self.loss != 'triplet_margin':
			return x
		else:
			return L2Norm()(x)

	def input_norm(self, x):
		'''
		flat = x.view(x.size(0), -1)
		mp = torch.mean(flat, dim=1)
		sp = torch.std(flat, dim=1) + 1e-7
		return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(
			-1).unsqueeze(1).expand_as(x)
		'''
		flat = x.view(x.size(0), x.size(1), -1)
		mp = torch.mean(flat, dim=2).unsqueeze(-1).unsqueeze(-1).expand_as(x)
		sp = torch.std(flat, dim=2).unsqueeze(-1).unsqueeze(-1).expand_as(x)
		if self.coordconv:
			mp[:, -1] = 0
			mp[:, -2] = 0
			sp[:, -1] = 1
			sp[:, -2] = 1
		return (x - mp) / sp.clamp(min=1e-7)


class ResBlock(nn.Module):
	def __init__(self, channels):
		super(ResBlock, self).__init__()
		self.resblock = nn.Sequential(
			nn.BatchNorm2d(channels, affine=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(channels, affine=False),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		return x + self.resblock(x)


class PatchDataset(torch.utils.data.Dataset):
	def __init__(self, image, pts, sigma, in_c=1, size=32, alpha=48, coordconv=False):
		super(PatchDataset, self).__init__()
		self.image = image
		self.pts = pts
		self.sigma = sigma
		self.alpha = alpha
		self.in_c = in_c
		self.size = size
		self.coordconv = coordconv

	def __getitem__(self, index):
		size = int(round(self.alpha * self.sigma[index]))
		pts = self.pts[index]
		if self.coordconv:
			xy_channel = get_coord_layer(self.size, pts, size)
		patch = cv2.getRectSubPix(self.image, (size, size), tuple(pts))
		patch = cv2.resize(patch, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
		if self.in_c == 1:
			patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
			patch = np.expand_dims(patch, axis=2)
		if self.coordconv:
			return torch.cat((ToTensor()(patch), xy_channel))
		else:
			return ToTensor()(patch)

	def __len__(self):
		return len(self.sigma)


def weights_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.orthogonal_(m.weight.data, gain=0.6)
		try:
			nn.init.constant(m.bias.data, 0.01)
		except:
			pass
	return


def create_loader(image, pts, sigma, in_c=1, size=32, bs=1024, num_workers=4, coordconv=False):
	return DataLoader(dataset=PatchDataset(image, pts, sigma, in_c, size, coordconv=coordconv),
	                  num_workers=num_workers,
	                  batch_size=bs,
	                  pin_memory=True,
	                  shuffle=False)


def get_desc(net, patch_loader, num_patches, device):
	descriptors = np.zeros((num_patches, 128), dtype=np.float32)
	descriptors_tensor = torch.from_numpy(descriptors).to(device)
	with torch.no_grad():
		bs_total = 0
		for patch in patch_loader:
			bs = len(patch)
			patch = patch.to(device)
			desc = net(patch)
			descriptors_tensor[bs_total:bs_total + bs] = desc.detach()
			bs_total += bs
	# torch.cuda.empty_cache()
	return descriptors_tensor.detach().cpu().numpy()


## For validation set
class PatchDataset_val(torch.utils.data.Dataset):
	def __init__(self, sub, logs, label, cam, in_c, size, alpha=48, coordconv=False):
		super(PatchDataset_val, self).__init__()
		# self.image_gray = image_gray
		self.sub = sub
		self.cam = cam
		self.logs = logs
		self.alpha = alpha
		self.label = label
		self.in_c = in_c
		self.size = size
		self.coordconv = coordconv
		## load faces
		## BDface
		CAMERAS = ['R10', 'R20', 'R30', 'R45']
		bosphorusDB_root = '/home/johnny/Datasets_hdd/BosphorusDB_prepared/20171117_rotated/'
		self.faces_data = {}
		for i, cam in enumerate(CAMERAS):
			self.faces_data[i] = dill.load(open('{}BosphorusDB_{}_20171117.pkl'.format(bosphorusDB_root, cam), "rb"))

	def __getitem__(self, index):
		cam = self.cam[index]
		sub = self.sub[index]
		face = self.faces_data[cam]['faces'][sub]
		log = self.logs[index]
		pts, sigma = log[:2], log[-1]
		size = int(round(self.alpha * sigma))
		if self.coordconv:
			h, w, _ = face.shape
			img_size = [w, h]
			nose = self.faces_data[cam]['pts'][sub][2]
			xy_channel = get_coord_layer(self.size, img_size, nose, pts, size)

		patch = cv2.getRectSubPix(face, (size, size), tuple(pts))
		patch = cv2.resize(patch, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
		# cv2.imwrite('./patch/{}.jpg'.format(index), patch)
		if self.in_c == 1:
			patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
			patch = np.expand_dims(patch, axis=2)
		if self.coordconv:
			return torch.cat((ToTensor()(patch), xy_channel))
		else:
			return ToTensor()(patch)

	def __len__(self):
		return len(self.sub)


def create_loader_val(feed_dict, in_c=1, size=32, bs=1024, num_workers=4, coordconv=False):
	sub = feed_dict['sub']
	logs = feed_dict['log']
	cam = feed_dict['cam']
	label = feed_dict['label']

	return DataLoader(dataset=PatchDataset_val(sub, logs, label, cam, in_c=in_c, size=size, coordconv=coordconv),
	                  num_workers=num_workers,
	                  batch_size=bs,
	                  pin_memory=True,
	                  shuffle=False,
	                  drop_last=False)


def get_desc_val(net, patch_loader, num_patches, device, self_aug=False):
	feature_dim = 128
	if self_aug:
		feature_dim *= 4
	descriptors = np.zeros((num_patches, feature_dim), dtype=np.float32)
	descriptors_tensor = torch.from_numpy(descriptors).to(device)
	with torch.no_grad():
		bs_total = 0
		for patch in patch_loader:
			bs = len(patch)
			patch = patch.to(device)
			desc = net(patch)
			descriptors_tensor[bs_total:bs_total + bs] = desc.detach()
			bs_total += bs
	return descriptors_tensor.detach().cpu().numpy()


## For training set
class PatchDataset_train(torch.utils.data.Dataset):
	def __init__(self, args, alpha=48):
		# sigma: 1.06 -> 3.54 (50*50, 67*67-> 170*170)
		super(PatchDataset_train, self).__init__()
		data_train = dill.load(open('patches_dataset_train.pkl', 'rb'))
		self.sub = data_train['sub']
		self.classes = len(self.sub)
		self.logs = data_train['log']
		self.alpha = alpha
		self.isAug = args.isAug
		self.which_aug = args.which_aug
		if self.isAug and self.which_aug == 1:
			self.len = self.classes * 2  ## 8 possible cases (random rotate 90 + flip)
		else:
			self.len = self.classes
		self.size = args.size
		self.n_channels = args.n_channels
		self.coordconv = args.coordconv
		## load faces
		## BDface
		CAMERAS = ['R10', 'R20', 'R30', 'R45']
		bosphorusDB_root = '/home/johnny/Datasets_hdd/BosphorusDB_prepared/20171117_rotated/'
		self.faces_data = {}
		for i, cam in enumerate(CAMERAS):
			self.faces_data[i] = dill.load(open('{}BosphorusDB_{}_20171117.pkl'.format(bosphorusDB_root, cam), "rb"))

	def __getitem__(self, index):
		group_id = index // self.classes
		index = index - group_id * self.classes

		## todo: update
		sub = self.sub[index].item()  # todo: update
		cams = np.arange(4)
		np.random.shuffle(cams)

		cam1, cam2 = cams[0], cams[1]
		face1, face2 = self.faces_data[cam1]['faces'][sub], self.faces_data[cam2]['faces'][sub]

		if self.coordconv:
			h1, w1, _ = face1.shape
			h2, w2, _ = face2.shape
			img_size1 = [w1, h1]
			img_size2 = [w2, h2]
			nose1, nose2 = self.faces_data[cam1]['pts'][sub][2], self.faces_data[cam2]['pts'][sub][2]

		log1, log2 = self.logs[index][cam1], self.logs[index][cam2]
		size1, size2 = int(round(self.alpha * log1[-1])), int(round(self.alpha * log2[-1]))

		# if self.isAug:
		# 	size1 = int(size1 * (random.random() * 0.6 + 0.7))  #0.7-1.3
		# 	size2 = int(size2 * (random.random() * 0.6 + 0.7))

		patch1 = cv2.getRectSubPix(face1, (size1, size1), tuple(log1[:2]))
		patch2 = cv2.getRectSubPix(face2, (size2, size2), tuple(log2[:2]))

		if self.isAug:
			if self.which_aug == 1:
				size1 = int(size1 * (random.random() * 0.6 + 0.7))
				size2 = int(size2 * (random.random() * 0.6 + 0.7))

			do_flip = random.random() > 0.5
			do_rot = random.random() > 0.5
			if do_rot:
				patch1 = np.rot90(patch1)
				patch2 = np.rot90(patch2)
			if do_flip:
				patch1 = cv2.flip(patch1, 1)
				patch2 = cv2.flip(patch2, 1)

		if self.coordconv:
			xy_channel1 = get_coord_layer(self.size, img_size1, nose1, log1[:2], size1)
			xy_channel2 = get_coord_layer(self.size, img_size2, nose2, log2[:2], size2)

		patch1 = cv2.resize(patch1, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
		patch2 = cv2.resize(patch2, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
		if self.n_channels == 1:
			patch1 = cv2.cvtColor(patch1, cv2.COLOR_BGR2GRAY)
			patch2 = cv2.cvtColor(patch2, cv2.COLOR_BGR2GRAY)
			patch1 = np.expand_dims(patch1, axis=2)
			patch2 = np.expand_dims(patch2, axis=2)
		elif self.n_channels == 3:
			pass
		if self.coordconv:
			tensor1 = torch.cat((ToTensor()(patch1), xy_channel1))
			tensor2 = torch.cat((ToTensor()(patch2), xy_channel2))
		else:
			tensor1 = ToTensor()(patch1)
			tensor2 = ToTensor()(patch2)
		label = torch.LongTensor(np.asarray([index]).reshape(1, 1))
		return tensor1, tensor2, label


	def __len__(self):
		return self.len

	def rotate_tensor(self, x, index):
		## x is a 3D tensor: C x H x W
		if index == 0:
			return x
		elif index == 1:
			return x.transpose(1, 2).flip(1)
		elif index == 2:
			return x.flip(1).flip(2)
		elif index == 3:
			return x.transpose(1, 2).flip(2)

class get_loader_train():
	def __init__(self, args, num_workers=8):
		self.dataset = PatchDataset_train(args)
		self.dataloader = DataLoader(dataset=self.dataset,
		                             num_workers=num_workers,
		                             batch_size=args.bs,
		                             pin_memory=True,
		                             shuffle=True,
		                             drop_last=True)
		self.train_iter = iter(self.dataloader)

	def next(self):
		try:
			data = next(self.train_iter)
		except:
			self.train_iter = iter(self.dataloader)
			data = next(self.train_iter)
		return data


def get_coord_layer(size, center_pt, rec_size):
		xx_channel = torch.arange(rec_size).repeat(1, rec_size, 1).type(torch.Tensor)
		yy_channel = torch.arange(rec_size).repeat(1, rec_size, 1).transpose(1, 2).type(torch.Tensor)

		## normalize to [-1, 1], maximum size is 170x170
		xx_channel = ((xx_channel - xx_channel[0][-1][-1] / 2) / 169.) * 2
		yy_channel = ((yy_channel - xx_channel[0][-1][-1] / 2) / 169.) * 2

		xy_channel = torch.cat((xx_channel, yy_channel))
		xy_channel = xy_channel.unsqueeze(0)

		## resize
		affine = torch.Tensor([1, 0, 0, 0, 1, 0]).view(-1, 2, 3)
		ones = torch.ones((1, 2, 3))
		theta = ones * affine
		grid = F.affine_grid(theta, torch.Size([1, 1, size, size]))
		xy_channel = F.grid_sample(xy_channel, grid)[0]

		return xy_channel