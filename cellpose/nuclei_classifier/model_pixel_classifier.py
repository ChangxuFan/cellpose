from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cellpose import models, transforms
from cellpose.core import assign_device, run_net
from cellpose.vit_sam import Transformer


class PixelClassifierNet(Transformer):
	"""CP-SAM encoder with a classifier-oriented output head."""

	def __init__(
		self,
		num_classes: int,
		predict_flows: bool = False,
		backbone: str = "vit_l",
		ps: int = 8,
		bsize: int = 256,
		rdrop: float = 0.4,
		dtype: torch.dtype = torch.float32,
	):
		if num_classes < 2:
			raise ValueError("num_classes must be >= 2")
		self.num_classes = int(num_classes)
		self.predict_flows = bool(predict_flows)
		nout = self.num_classes + (3 if self.predict_flows else 0)
		super().__init__(
			backbone=backbone,
			ps=ps,
			nout=nout,
			bsize=bsize,
			rdrop=rdrop,
			dtype=dtype,
		)


def _load_matching_state_dict(net: torch.nn.Module, model_path: str | Path, device: torch.device):
	"""Load only matching-shape weights so we can reuse encoder with a new head."""
	state = torch.load(model_path, map_location=device, weights_only=True)
	if any(k.startswith("module.") for k in state):
		state = {k.replace("module.", "", 1): v for k, v in state.items()}

	current = net.state_dict()
	matched = {
		k: v
		for k, v in state.items()
		if k in current and current[k].shape == v.shape
	}
	current.update(matched)
	net.load_state_dict(current, strict=False)


class CellposePixelClassifier:
	"""Model wrapper for pixel-wise category prediction, with optional flow channels."""

	def __init__(
		self,
		num_classes: int,
		predict_flows: bool = False,
		gpu: bool = False,
		pretrained_model: str = "cpsam",
		device: torch.device | None = None,
		use_bfloat16: bool = True,
	):
		self.device = assign_device(gpu=gpu)[0] if device is None else device
		if torch.cuda.is_available():
			self.gpu = self.device.type == "cuda"
		elif torch.backends.mps.is_available():
			self.gpu = self.device.type == "mps"
		else:
			self.gpu = False

		dtype = torch.bfloat16 if use_bfloat16 else torch.float32
		self.net = PixelClassifierNet(
			num_classes=num_classes,
			predict_flows=predict_flows,
			dtype=dtype,
		).to(self.device)
		self.num_classes = int(num_classes)
		self.predict_flows = bool(predict_flows)

		if pretrained_model:
			path = str(pretrained_model)
			if not os.path.exists(path):
				if pretrained_model in models.MODEL_NAMES:
					path = str(models.MODEL_DIR / pretrained_model)
				else:
					path = str(models.MODEL_DIR / "cpsam")
			if not os.path.exists(path) and os.path.basename(path) == "cpsam":
				models.cache_CPSAM_model_path()
			_load_matching_state_dict(self.net, path, self.device)

	def eval(
		self,
		x,
		batch_size: int = 8,
		channel_axis: int | None = None,
		z_axis: int | None = None,
		normalize=True,
		invert: bool = False,
		rescale: float | None = None,
		tile_overlap: float = 0.1,
		bsize: int = 256,
	):
		if isinstance(x, list):
			class_maps, prob_maps, flow_maps, styles = [], [], [], []
			for xi in x:
				cm, pm, fm, st = self.eval(
					xi,
					batch_size=batch_size,
					channel_axis=channel_axis,
					z_axis=z_axis,
					normalize=normalize,
					invert=invert,
					rescale=rescale,
					tile_overlap=tile_overlap,
					bsize=bsize,
				)
				class_maps.append(cm)
				prob_maps.append(pm)
				flow_maps.append(fm)
				styles.append(st)
			return class_maps, prob_maps, flow_maps, styles

		img = transforms.convert_image(x, channel_axis=channel_axis, z_axis=z_axis, do_3D=False)
		if img.ndim < 4:
			img = img[np.newaxis, ...]

		normalize_params = models.normalize_default.copy()
		if isinstance(normalize, dict):
			normalize_params.update(normalize)
		elif isinstance(normalize, bool):
			normalize_params["normalize"] = normalize
			normalize_params["invert"] = invert
		else:
			raise ValueError("normalize parameter must be a bool or a dict")

		if normalize_params.get("normalize", True):
			img = transforms.normalize_img(img, **normalize_params)

		yf, styles = run_net(
			self.net,
			img,
			batch_size=batch_size,
			tile_overlap=tile_overlap,
			bsize=bsize,
			rsz=rescale,
		)

		logits = yf[..., : self.num_classes]
		probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy().astype(np.float32)
		classes = np.argmax(probs, axis=-1).astype(np.uint16)

		if classes.shape[0] == 1:
			classes = classes[0]
			probs = probs[0]

		flows = None
		if self.predict_flows:
			flow_yx = yf[..., self.num_classes : self.num_classes + 2]
			cellprob = yf[..., self.num_classes + 2]
			flows = {
				"dP": flow_yx,
				"cellprob": cellprob,
			}

		return classes, probs, flows, styles


def initialize_classifier_model(
	num_classes: int,
	predict_flows: bool = False,
	gpu: bool = False,
	pretrained_model: str = "cpsam",
	device: torch.device | None = None,
	use_bfloat16: bool = True,
) -> CellposePixelClassifier:
	"""Factory helper for notebook and CLI usage."""
	return CellposePixelClassifier(
		num_classes=num_classes,
		predict_flows=predict_flows,
		gpu=gpu,
		pretrained_model=pretrained_model,
		device=device,
		use_bfloat16=use_bfloat16,
	)
