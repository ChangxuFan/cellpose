from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from cellpose import dynamics, io


def _load_label_array(label_source) -> np.ndarray:
	if isinstance(label_source, np.ndarray):
		arr = label_source
	else:
		path = Path(label_source)
		arr = io.imread(str(path))
	arr = np.asarray(arr)
	if arr.ndim > 2:
		arr = np.squeeze(arr)
	if arr.ndim != 2:
		raise ValueError(f"Expected 2D label map, got shape {arr.shape}")
	return arr.astype(np.int32)


def ensure_class_maps(train_labels) -> list[np.ndarray]:
	labels = [_load_label_array(lbl) for lbl in train_labels]
	if len(labels) == 0:
		raise ValueError("No labels were provided")
	for idx, lbl in enumerate(labels):
		if np.any(lbl < 0):
			raise ValueError(f"Negative class ids found in label index {idx}")
	return labels


def remap_class_ids(class_maps: list[np.ndarray]) -> tuple[list[np.ndarray], pd.DataFrame]:
	"""Map class ids to contiguous values with 0 reserved for background."""
	all_ids = sorted({int(v) for arr in class_maps for v in np.unique(arr)})
	non_bg = [v for v in all_ids if v != 0]
	mapping = {0: 0}
	for i, cid in enumerate(non_bg, start=1):
		mapping[cid] = i

	max_in = max(mapping.keys()) if mapping else 0
	lut = np.zeros(max_in + 1, dtype=np.int32)
	for k, v in mapping.items():
		lut[k] = v

	remapped = []
	for arr in class_maps:
		if arr.max() > max_in:
			out = np.zeros_like(arr, dtype=np.int32)
			for k, v in mapping.items():
				out[arr == k] = v
		else:
			out = lut[arr]
		remapped.append(out)

	map_df = pd.DataFrame(
		[{"source_id": k, "train_id": v} for k, v in sorted(mapping.items(), key=lambda kv: kv[1])]
	)
	return remapped, map_df


def infer_instances_from_types(class_map: np.ndarray) -> np.ndarray:
	"""Connected-components instance labels per class id, excluding background (0)."""
	class_map = np.asarray(class_map)
	out = np.zeros_like(class_map, dtype=np.int32)
	next_id = 1
	for cls_id in sorted(int(v) for v in np.unique(class_map) if int(v) > 0):
		cc, n_cc = ndimage.label(class_map == cls_id)
		if n_cc == 0:
			continue
		cc = cc.astype(np.int32)
		mask = cc > 0
		cc[mask] += next_id - 1
		out[mask] = cc[mask]
		next_id += int(n_cc)
	return out


def build_training_labels(
	class_maps: list[np.ndarray],
	compute_flows: bool,
	device,
) -> list[np.ndarray]:
	"""
	Return labels for classifier training:
	- without flows: [class]
	- with flows: [class, cellprob, flowY, flowX]
	"""
	if not compute_flows:
		return [arr[np.newaxis, ...].astype(np.float32) for arr in class_maps]

	instance_masks = [infer_instances_from_types(arr) for arr in class_maps]
	flow_labels = dynamics.labels_to_flows(instance_masks, files=None, device=device)

	out = []
	for cls, fl in zip(class_maps, flow_labels, strict=False):
		# fl is expected as [instance_mask, cellprob, flowY, flowX]
		if fl.shape[0] < 4:
			raise ValueError("Flow labels did not contain expected channels")
		merged = np.stack(
			[
				cls.astype(np.float32),
				fl[1].astype(np.float32),
				fl[2].astype(np.float32),
				fl[3].astype(np.float32),
			],
			axis=0,
		)
		out.append(merged)
	return out
