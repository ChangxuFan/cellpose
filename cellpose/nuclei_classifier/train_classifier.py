from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cellpose import io, models
from cellpose.train import _reshape_norm
from cellpose.transforms import random_rotate_and_resize

from .data_classifier import build_training_labels, ensure_class_maps, remap_class_ids


def _loss_fn_class(lbl: torch.Tensor, y: torch.Tensor, num_classes: int, class_weights=None):
	criterion = nn.CrossEntropyLoss(reduction="mean", weight=class_weights)
	target = lbl[:, 0].long().clamp_(min=0, max=num_classes - 1)
	return criterion(y[:, :num_classes], target)


def _loss_fn_flow(lbl: torch.Tensor, y: torch.Tensor, num_classes: int):
	criterion = nn.MSELoss(reduction="mean")
	criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
	pred = y[:, num_classes : num_classes + 3]
	veci = 5.0 * lbl[:, 2:4]
	loss = criterion(pred[:, :2], veci)
	loss /= 2.0
	loss2 = criterion2(pred[:, 2], (lbl[:, 1] > 0.5).to(y.dtype))
	return loss + loss2


def _get_batch(inds, data=None, labels=None, files=None, labels_files=None, normalize_params=None):
	if data is None:
		imgs = [io.imread(files[i]) for i in inds]
		imgs = _reshape_norm(imgs, normalize_params=normalize_params)
		lbls = [io.imread(labels_files[i]) for i in inds]
	else:
		imgs = [data[i] for i in inds]
		lbls = [labels[i] for i in inds]
	return imgs, lbls


def _normalize_data(train_data, test_data, channel_axis, normalize):
	if isinstance(normalize, dict):
		normalize_params = {**models.normalize_default, **normalize}
	elif not isinstance(normalize, bool):
		raise ValueError("normalize parameter must be a bool or a dict")
	else:
		normalize_params = models.normalize_default
		normalize_params["normalize"] = normalize

	train_data = _reshape_norm(train_data, channel_axis=channel_axis, normalize_params=normalize_params)
	if test_data is not None:
		test_data = _reshape_norm(test_data, channel_axis=channel_axis, normalize_params=normalize_params)
	return train_data, test_data, normalize_params


def train_classifier(
	net,
	train_data=None,
	train_labels=None,
	train_files=None,
	train_labels_files=None,
	train_probs=None,
	test_data=None,
	test_labels=None,
	test_files=None,
	test_labels_files=None,
	test_probs=None,
	channel_axis=None,
	load_files=True,
	batch_size=1,
	learning_rate=1e-5,
	n_epochs=100,
	weight_decay=0.1,
	normalize=True,
	compute_flows=False,
	flow_loss_weight=1.0,
	save_path=None,
	save_every=100,
	save_each=False,
	nimg_per_epoch=None,
	nimg_test_per_epoch=None,
	rescale=False,
	scale_range=None,
	do_flip=True,
	bsize=256,
	model_name=None,
	class_weights=None,
	write_class_map=True,
):
	"""Train a pixel classifier model using class maps and optional flow supervision."""
	device = net.device
	original_net_dtype = net.dtype
	if net.dtype == torch.bfloat16:
		net.dtype = torch.float32

	if train_data is None:
		if train_files is None:
			raise ValueError("Either train_data or train_files must be provided")
		if train_labels_files is None:
			raise ValueError("train_labels_files must be provided when using train_files")
		if load_files:
			train_data = [io.imread(train_files[i]) for i in range(len(train_files))]
			train_labels = [io.imread(train_labels_files[i]) for i in range(len(train_labels_files))]
		else:
			train_data = None
			train_labels = None

	if train_labels is None:
		raise ValueError("train_labels are required for train_classifier")

	if test_data is None and test_files is not None and test_labels_files is not None and load_files:
		test_data = [io.imread(test_files[i]) for i in range(len(test_files))]
		test_labels = [io.imread(test_labels_files[i]) for i in range(len(test_labels_files))]

	class_maps_train = ensure_class_maps(train_labels)
	class_maps_train, class_map_df = remap_class_ids(class_maps_train)
	num_classes = int(max(int(v) for v in class_map_df["train_id"].tolist()) + 1)
	if num_classes != getattr(net, "num_classes", num_classes):
		raise ValueError(
			f"Network num_classes ({getattr(net, 'num_classes', None)}) does not match training labels ({num_classes})"
		)

	train_labels = build_training_labels(class_maps_train, compute_flows=compute_flows, device=device)

	if test_labels is not None:
		class_maps_test = ensure_class_maps(test_labels)
		# Reuse train mapping and clamp unknown ids to background for stable evaluation.
		map_dict = dict(zip(class_map_df["source_id"], class_map_df["train_id"], strict=False))
		class_maps_test = [np.vectorize(lambda z: map_dict.get(int(z), 0))(arr).astype(np.int32) for arr in class_maps_test]
		test_labels = build_training_labels(class_maps_test, compute_flows=compute_flows, device=device)

	if train_data is not None:
		train_data, test_data, normalize_params = _normalize_data(
			train_data,
			test_data,
			channel_axis=channel_axis,
			normalize=normalize,
		)
	else:
		normalize_params = {**models.normalize_default, "normalize": bool(normalize)}

	if class_weights is not None and isinstance(class_weights, (list, tuple, np.ndarray)):
		class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
	else:
		class_weights = None

	nimg = len(train_data) if train_data is not None else len(train_files)
	nimg_test = len(test_data) if test_data is not None else (len(test_files) if test_files is not None else 0)
	nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
	nimg_test_per_epoch = nimg_test if nimg_test_per_epoch is None else nimg_test_per_epoch

	train_probs = np.ones(nimg, dtype=np.float64) / nimg if train_probs is None else np.asarray(train_probs, dtype=np.float64)
	train_probs /= train_probs.sum()
	if nimg_test > 0:
		test_probs = np.ones(nimg_test, dtype=np.float64) / nimg_test if test_probs is None else np.asarray(test_probs, dtype=np.float64)
		test_probs /= test_probs.sum()

	scale_range = 0.5 if scale_range is None else scale_range
	LR = np.linspace(0, learning_rate, 10)
	LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))
	if n_epochs > 300:
		LR = LR[:-100]
		for _ in range(10):
			LR = np.append(LR, LR[-1] / 2 * np.ones(10))
	elif n_epochs > 99:
		LR = LR[:-50]
		for _ in range(10):
			LR = np.append(LR, LR[-1] / 2 * np.ones(5))

	optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

	t0 = time.time()
	model_name = f"cellpose_pixel_classifier_{t0}" if model_name is None else model_name
	save_path = Path.cwd() if save_path is None else Path(save_path)
	(save_path / "models").mkdir(exist_ok=True)
	filename = save_path / "models" / model_name

	if write_class_map:
		class_map_df.to_csv(save_path / "models" / f"{model_name}_class_id_map.tsv", sep="\t", index=False)

	train_losses = np.zeros(n_epochs, dtype=np.float32)
	test_losses = np.zeros(n_epochs, dtype=np.float32)

	for iepoch in range(n_epochs):
		np.random.seed(iepoch)
		rperm = (
			np.random.choice(np.arange(0, nimg), size=(nimg_per_epoch,), p=train_probs)
			if nimg != nimg_per_epoch
			else np.random.permutation(np.arange(0, nimg))
		)

		for param_group in optimizer.param_groups:
			param_group["lr"] = float(LR[iepoch])

		net.train()
		for k in range(0, nimg_per_epoch, batch_size):
			inds = rperm[k : min(k + batch_size, nimg_per_epoch)]
			imgs, lbls = _get_batch(
				inds,
				data=train_data,
				labels=train_labels,
				files=train_files,
				labels_files=train_labels_files,
				normalize_params=normalize_params,
			)

			rsc = np.ones(len(inds), dtype=np.float32)
			imgi, lbl = random_rotate_and_resize(
				imgs,
				Y=lbls,
				rescale=rsc if rescale else None,
				scale_range=scale_range,
				do_flip=do_flip,
				xy=(bsize, bsize),
			)[:2]

			X = torch.from_numpy(imgi).to(device)
			lbl = torch.from_numpy(lbl).to(device)

			with torch.autocast(device_type=device.type, dtype=net.dtype):
				y = net(X)[0]
			loss = _loss_fn_class(lbl, y, num_classes=num_classes, class_weights=class_weights)
			if compute_flows:
				if y.shape[1] < num_classes + 3:
					raise ValueError("Model output does not include flow channels, but compute_flows=True")
				loss = loss + float(flow_loss_weight) * _loss_fn_flow(lbl, y, num_classes=num_classes)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			train_losses[iepoch] += float(loss.item()) * len(imgi)

		train_losses[iepoch] /= max(1, nimg_per_epoch)

		if (iepoch == 5 or iepoch % 10 == 0) and nimg_test > 0 and test_data is not None and test_labels is not None:
			lavgt = 0.0
			np.random.seed(42)
			rperm_t = (
				np.random.choice(np.arange(0, nimg_test), size=(nimg_test_per_epoch,), p=test_probs)
				if nimg_test != nimg_test_per_epoch
				else np.random.permutation(np.arange(0, nimg_test))
			)
			net.eval()
			for k in range(0, len(rperm_t), batch_size):
				inds = rperm_t[k : k + batch_size]
				imgs, lbls = _get_batch(
					inds,
					data=test_data,
					labels=test_labels,
					files=test_files,
					labels_files=test_labels_files,
					normalize_params=normalize_params,
				)
				imgi, lbl = random_rotate_and_resize(
					imgs,
					Y=lbls,
					rescale=None,
					scale_range=scale_range,
					do_flip=do_flip,
					xy=(bsize, bsize),
				)[:2]
				X = torch.from_numpy(imgi).to(device)
				lbl = torch.from_numpy(lbl).to(device)
				with torch.no_grad():
					with torch.autocast(device_type=device.type, dtype=net.dtype):
						y = net(X)[0]
					loss = _loss_fn_class(lbl, y, num_classes=num_classes, class_weights=class_weights)
					if compute_flows:
						loss = loss + float(flow_loss_weight) * _loss_fn_flow(lbl, y, num_classes=num_classes)
				lavgt += float(loss.item()) * len(imgi)
			test_losses[iepoch] = lavgt / max(1, len(rperm_t))

		if iepoch == n_epochs - 1 or (iepoch % save_every == 0 and iepoch != 0):
			if save_each and iepoch != n_epochs - 1:
				filename0 = str(filename) + f"_epoch_{iepoch:04d}"
			else:
				filename0 = filename
			net.save_model(filename0)

	net.save_model(filename)
	if original_net_dtype != torch.float32:
		net.dtype = original_net_dtype

	return filename, train_losses, test_losses
