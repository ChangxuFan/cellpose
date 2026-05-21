from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cellpose import io


def _to_u8(img: np.ndarray) -> np.ndarray:
	arr = np.asarray(img)
	if arr.ndim == 3:
		arr = arr[..., 0]
	if arr.dtype == np.uint8:
		return arr
	arr = arr.astype(np.float32)
	low, high = np.percentile(arr, [1, 99])
	if high <= low:
		high = low + 1.0
	arr = np.clip((arr - low) / (high - low), 0.0, 1.0)
	return (arr * 255).astype(np.uint8)


def _class_palette(max_cls: int) -> np.ndarray:
	"""Deterministic RGB palette for class ids; class 0 is black."""
	pal = np.zeros((max_cls + 1, 3), dtype=np.uint8)
	if max_cls <= 0:
		return pal
	for c in range(1, max_cls + 1):
		# Simple hash-like color generation with good spread.
		pal[c, 0] = (53 * c + 47) % 256
		pal[c, 1] = (97 * c + 131) % 256
		pal[c, 2] = (193 * c + 71) % 256
	return pal


def _colorize_classes(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
	mask = np.asarray(mask, dtype=np.int32)
	if mask.min() < 0:
		raise ValueError("Class masks must be non-negative integers")
	max_needed = int(mask.max())
	if max_needed >= len(palette):
		extra = _class_palette(max_needed)
		palette = extra
	return palette[mask]


def _make_error_panel(gt_mask: np.ndarray, pred_mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
	"""
	Class-aware error panel:
	- correct pixels: black
	- wrong pixels: color of predicted class
	"""
	err = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
	wrong = gt_mask != pred_mask
	pred_col = _colorize_classes(pred_mask, palette)
	err[wrong] = pred_col[wrong]
	return err


def masks_cmp(
	gt_dir: str | Path,
	pred_dir: str | Path,
	cmp_dir: str | Path,
	max_images: int | None = None,
) -> dict[str, str]:
	"""
	Compare ground-truth and predicted masks for matching image tiles.

	Requirements:
	- `gt_dir` contains image tiles (`*.tif`) and corresponding `*_masks.tif`.
	- `pred_dir` contains matching image tiles and corresponding predicted `*_masks.tif`.

	Returns:
	- Mapping from tile stem to comparison PNG path.
	"""
	gt_dir = Path(gt_dir)
	pred_dir = Path(pred_dir)
	cmp_dir = Path(cmp_dir)
	cmp_dir.mkdir(parents=True, exist_ok=True)

	gt_images = sorted([p for p in gt_dir.glob("*.tif") if not p.name.endswith("_masks.tif")])
	if len(gt_images) == 0:
		raise ValueError(f"No image tiles found in {gt_dir}")

	if max_images is not None and max_images > 0:
		gt_images = gt_images[:max_images]

	font = ImageFont.load_default()
	out: dict[str, str] = {}
	metrics_dir = cmp_dir / "metrics"
	metrics_dir.mkdir(parents=True, exist_ok=True)

	global_conf: np.ndarray | None = None
	per_image_rows: list[dict] = []

	for img_path in gt_images:
		stem = img_path.stem
		gt_mask_path = gt_dir / f"{stem}_masks.tif"
		pred_img_path = pred_dir / f"{stem}.tif"
		pred_mask_path = pred_dir / f"{stem}_masks.tif"

		if not gt_mask_path.exists() or not pred_img_path.exists() or not pred_mask_path.exists():
			continue

		img = io.imread(str(img_path))
		img_u8 = _to_u8(img)

		gt_mask = io.imread(str(gt_mask_path)).astype(np.int32)
		pred_mask = io.imread(str(pred_mask_path)).astype(np.int32)

		if gt_mask.shape != pred_mask.shape:
			raise ValueError(f"Mask shape mismatch for {stem}: gt={gt_mask.shape}, pred={pred_mask.shape}")

		max_cls = int(max(gt_mask.max(), pred_mask.max()))
		palette = _class_palette(max_cls)
		gt_cls = _colorize_classes(gt_mask, palette)
		pred_cls = _colorize_classes(pred_mask, palette)
		err_panel = _make_error_panel(gt_mask, pred_mask, palette)

		if global_conf is None or global_conf.shape[0] <= max_cls:
			new_size = max_cls + 1
			new_conf = np.zeros((new_size, new_size), dtype=np.int64)
			if global_conf is not None:
				new_conf[: global_conf.shape[0], : global_conf.shape[1]] = global_conf
			global_conf = new_conf

		conf = np.zeros((max_cls + 1, max_cls + 1), dtype=np.int64)
		gt_flat = gt_mask.ravel()
		pred_flat = pred_mask.ravel()
		for g, p in zip(gt_flat, pred_flat):
			conf[g, p] += 1
			global_conf[g, p] += 1

		pix_acc = float((pred_mask == gt_mask).mean())
		fg = gt_mask > 0
		fg_pix_acc = float((pred_mask[fg] == gt_mask[fg]).mean()) if np.any(fg) else np.nan

		per_class_lines = []
		for c in range(max_cls + 1):
			tp = conf[c, c]
			fp = conf[:, c].sum() - tp
			fn = conf[c, :].sum() - tp
			den = tp + fp + fn
			iou = float(tp / den) if den > 0 else np.nan
			dice_den = 2 * tp + fp + fn
			dice = float((2 * tp) / dice_den) if dice_den > 0 else np.nan
			support = int(conf[c, :].sum())
			per_class_lines.append(
				f"class {c}: iou={iou:.6f}, dice={dice:.6f}, support={support}"
			)

		per_image_rows.append(
			{
				"tile": stem,
				"pixel_acc": pix_acc,
				"fg_pixel_acc": fg_pix_acc,
				"fg_fraction": float(fg.mean()),
				"max_class_id": max_cls,
			}
		)

		img_report = [
			f"tile: {stem}",
			f"pixel_acc: {pix_acc:.6f}",
			f"fg_pixel_acc: {fg_pix_acc:.6f}",
			f"fg_fraction: {float(fg.mean()):.6f}",
			"per_class:",
			*per_class_lines,
		]
		(metrics_dir / f"{stem}.txt").write_text("\n".join(img_report) + "\n")

		panels = [
			("image", np.stack([img_u8, img_u8, img_u8], axis=-1)),
			("gt_class", gt_cls),
			("pred_class", pred_cls),
			("error_pred_class", err_panel),
		]

		title_h = 26
		pad = 8
		w = panels[0][1].shape[1]
		h = panels[0][1].shape[0]
		canvas = Image.new("RGB", (len(panels) * w + (len(panels) - 1) * pad, h + title_h), (255, 255, 255))
		draw = ImageDraw.Draw(canvas)

		for i, (name, arr) in enumerate(panels):
			x0 = i * (w + pad)
			draw.text((x0 + 4, 4), name, fill=(0, 0, 0), font=font)
			canvas.paste(Image.fromarray(arr), (x0, title_h))

		out_path = cmp_dir / f"{stem}__gt_vs_pred.png"
		canvas.save(out_path)
		out[stem] = str(out_path)

	if len(out) == 0:
		raise ValueError("No comparable image/mask pairs found between gt_dir and pred_dir")

	assert global_conf is not None
	global_lines = []
	mean_pix = float(np.mean([r["pixel_acc"] for r in per_image_rows]))
	mean_fg = float(np.nanmean([r["fg_pixel_acc"] for r in per_image_rows]))
	global_lines.append(f"n_images: {len(per_image_rows)}")
	global_lines.append(f"mean_pixel_acc: {mean_pix:.6f}")
	global_lines.append(f"mean_fg_pixel_acc: {mean_fg:.6f}")
	global_lines.append("per_class:")
	for c in range(global_conf.shape[0]):
		tp = global_conf[c, c]
		fp = global_conf[:, c].sum() - tp
		fn = global_conf[c, :].sum() - tp
		den = tp + fp + fn
		iou = float(tp / den) if den > 0 else np.nan
		dice_den = 2 * tp + fp + fn
		dice = float((2 * tp) / dice_den) if dice_den > 0 else np.nan
		support = int(global_conf[c, :].sum())
		global_lines.append(
			f"class {c}: iou={iou:.6f}, dice={dice:.6f}, support={support}"
		)

	(cmp_dir / "metrics_overall.txt").write_text("\n".join(global_lines) + "\n")

	return out
