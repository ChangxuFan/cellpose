from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import shapely
import tifffile
from PIL import Image, ImageDraw
from shapely.geometry import Polygon


XENIUM_PIXEL_SIZE = 0.2125  # microns per pixel in Xenium outputs


def _read_parquet_robust(path: str | Path) -> pd.DataFrame:
	"""Read parquet with a tolerant engine fallback order."""
	errs = []
	for engine in ["fastparquet", "pyarrow", None]:
		try:
			if engine is None:
				return pd.read_parquet(path)
			return pd.read_parquet(path, engine=engine)
		except Exception as exc:
			errs.append(f"{engine}: {exc}")
	joined = " | ".join(errs)
	raise RuntimeError(f"Unable to read parquet file {path}. Errors: {joined}")


def _load_cell_groups(cell_groups: str | Path | pd.DataFrame) -> pd.DataFrame:
	if isinstance(cell_groups, pd.DataFrame):
		df = cell_groups.copy()
	else:
		path = Path(cell_groups)
		suffix = path.suffix.lower()
		if suffix == ".csv":
			df = pd.read_csv(path)
		elif suffix in {".tsv", ".txt"}:
			df = pd.read_csv(path, sep="\t")
		else:
			raise ValueError(f"Unsupported cell_groups format: {suffix}")

	required = {"cell_id", "group"}
	missing = required.difference(df.columns)
	if missing:
		raise ValueError(f"cell_groups is missing required columns: {sorted(missing)}")

	out = df[["cell_id", "group"]].copy()
	out["cell_id"] = out["cell_id"].astype(str)
	out["group"] = out["group"].astype(str)
	out = out.drop_duplicates(subset=["cell_id"], keep="first")
	return out


def _coords_to_polygon(gr: pd.DataFrame) -> Polygon | None:
	if "vertex_i" in gr.columns:
		gr = gr.sort_values("vertex_i")
	coords = gr[["x", "y"]].to_numpy()
	if coords.shape[0] < 3:
		return None
	poly = Polygon(coords)
	if not poly.is_valid:
		poly = shapely.make_valid(poly)
		if poly.geom_type == "Polygon":
			return poly
		if poly.geom_type == "MultiPolygon":
			return max(poly.geoms, key=lambda g: g.area)
		return None
	return poly


def _build_nucleus_polygons(nucleus_df: pd.DataFrame) -> dict[str, Polygon]:
	df = nucleus_df.copy()
	df["cell_id_str"] = df["cell_id"].astype(str)
	out: dict[str, Polygon] = {}
	for cell_id, gr in df.groupby("cell_id_str", sort=False):
		poly = _coords_to_polygon(gr)
		if poly is not None and poly.area > 0:
			out[cell_id] = poly
	return out


def _weighted_tile_sample(tile_df: pd.DataFrame, n_crops: int, seed: int) -> pd.DataFrame:
	if tile_df.empty:
		raise ValueError("No candidate tiles with nuclei were found")

	stable = tile_df.sort_values(["score", "known_fraction", "n_cells"], ascending=[False, False, False]).copy()
	# Prefer tiles with enough labeled cells.
	preferred = stable[stable["known_fraction"] >= 0.5]
	pool = preferred if len(preferred) >= min(n_crops, 5) else stable

	if len(pool) <= n_crops:
		return pool.reset_index(drop=True)

	rng = np.random.default_rng(seed)
	weights = pool["score"].to_numpy(dtype=np.float64)
	weights = np.clip(weights, 1e-9, None)
	probs = weights / weights.sum()
	idx = rng.choice(len(pool), size=n_crops, replace=False, p=probs)
	return pool.iloc[np.sort(idx)].reset_index(drop=True)


def _resolve_zarr_array(obj: Any):
	"""Return the first array-like object with shape/ndim from a zarr Group/Array tree."""
	if hasattr(obj, "shape") and hasattr(obj, "ndim"):
		return obj
	if not hasattr(obj, "keys"):
		raise ValueError("Unable to resolve zarr array from object")

	for key in sorted(obj.keys()):
		child = obj[key]
		if hasattr(child, "shape") and hasattr(child, "ndim"):
			return child
		if hasattr(child, "keys"):
			try:
				return _resolve_zarr_array(child)
			except ValueError:
				continue
	raise ValueError("No array node found in zarr group")


def _read_tile_mip(
	series,
	axes: str,
	bbox_xy: tuple[int, int, int, int],
	arr=None,
) -> np.ndarray:
	x0, y0, x1, y1 = bbox_xy
	index = []
	kept_axes = []
	for ax in axes:
		if ax == "Y":
			index.append(slice(y0, y1))
			kept_axes.append("Y")
		elif ax == "X":
			index.append(slice(x0, x1))
			kept_axes.append("X")
		elif ax == "Z":
			index.append(slice(None))
			kept_axes.append("Z")
		else:
			index.append(0)

	if arr is not None:
		patch = np.asarray(arr[tuple(index)])
	else:
		patch = np.asarray(series.asarray(key=tuple(index)))

	if "Z" in kept_axes:
		z_axis = kept_axes.index("Z")
		patch = np.max(patch, axis=z_axis)

	patch = np.squeeze(patch)
	if patch.ndim != 2:
		raise ValueError(f"Expected 2D MIP patch, got shape {patch.shape}")
	return patch


def _draw_tile_masks(
	cell_polys: pd.DataFrame,
	x0: int,
	y0: int,
	x1: int,
	y1: int,
) -> np.ndarray:
	w = x1 - x0
	h = y1 - y0
	canvas = Image.new("I", (w, h), 0)
	draw = ImageDraw.Draw(canvas)

	for row in cell_polys.itertuples(index=False):
		poly = row.polygon
		gid = int(row.group_id)
		ext = np.asarray(poly.exterior.coords)
		pts = [(float(x - x0), float(y - y0)) for x, y in ext]
		draw.polygon(pts, fill=gid)

	return np.asarray(canvas, dtype=np.uint16)


def _pad_to_square(arr: np.ndarray, n_pixel: int) -> np.ndarray:
	out = np.zeros((n_pixel, n_pixel), dtype=arr.dtype)
	h, w = arr.shape
	out[:h, :w] = arr
	return out


def training_data_prep(
	xenium_dir,
	cell_groups,
	n_pixel: int = 512,
	n_crops: int = 50,
	seed: int = 42,
	outdir=None,
	root=None,
):
	"""
	Prepare tile-level training images and group masks from Xenium morphology + nuclei boundaries.

	Outputs:
	- {root}_{x}_{y}.tif: MIP intensity crop across all Z planes.
	- {root}_{x}_{y}_masks.tif: integer mask image with group ids.
	- {root}_group_to_id.tsv: mapping table with columns (group, id).
	"""
	if outdir is None:
		raise ValueError("outdir must be provided")
	if root is None:
		raise ValueError("root must be provided")
	if n_pixel <= 0:
		raise ValueError("n_pixel must be > 0")
	if n_crops <= 0:
		raise ValueError("n_crops must be > 0")

	xenium_dir = Path(xenium_dir)
	outdir = Path(outdir)
	outdir.mkdir(parents=True, exist_ok=True)

	nucleus_path = xenium_dir / "nucleus_boundaries.parquet"
	morph_path = xenium_dir / "morphology.ome.tif"
	for p in [nucleus_path, morph_path]:
		if not p.exists():
			raise FileNotFoundError(f"Required file not found: {p}")

	nucleus_df = _read_parquet_robust(nucleus_path)
	nucleus_df["x"] = nucleus_df["vertex_x"] / XENIUM_PIXEL_SIZE
	nucleus_df["y"] = nucleus_df["vertex_y"] / XENIUM_PIXEL_SIZE

	group_df = _load_cell_groups(cell_groups)
	group_map = dict(zip(group_df["cell_id"], group_df["group"], strict=False))

	nucleus_polys = _build_nucleus_polygons(nucleus_df)
	if not nucleus_polys:
		raise ValueError("No valid nucleus polygons were found")

	cell_ids_all = list(nucleus_polys.keys())
	total_cells = len(cell_ids_all)
	covered_cells = sum(1 for cid in cell_ids_all if cid in group_map)
	not_covered_cells = total_cells - covered_cells
	not_covered_pct = (100.0 * not_covered_cells / total_cells) if total_cells else 0.0
	print(
		f"Cells covered by cell_groups: {covered_cells}/{total_cells}. "
		f"Not covered: {not_covered_cells} ({not_covered_pct:.2f}%)."
	)

	unknown_label = "unknown"
	known_groups = sorted({g for g in group_df["group"].astype(str).tolist() if g != unknown_label})
	group_to_id = {g: i + 1 for i, g in enumerate(known_groups)}
	group_to_id[unknown_label] = len(known_groups) + 1

	map_rows = [{"group": "background", "id": 0}]
	map_rows.extend({"group": g, "id": gid} for g, gid in group_to_id.items())
	map_df = pd.DataFrame(map_rows)
	map_path = outdir / f"{root}_group_to_id.tsv"
	map_df.to_csv(map_path, sep="\t", index=False)

	records = []
	for cid, poly in nucleus_polys.items():
		grp = group_map.get(cid, unknown_label)
		gid = group_to_id[grp]
		cx, cy = poly.centroid.x, poly.centroid.y
		minx, miny, maxx, maxy = poly.bounds
		records.append(
			{
				"cell_id": cid,
				"group": grp,
				"group_id": gid,
				"known": grp != unknown_label,
				"cx": cx,
				"cy": cy,
				"minx": minx,
				"miny": miny,
				"maxx": maxx,
				"maxy": maxy,
				"polygon": poly,
			}
		)

	poly_df = pd.DataFrame(records)
	poly_df["tile_x"] = (poly_df["cx"] // n_pixel).astype(int)
	poly_df["tile_y"] = (poly_df["cy"] // n_pixel).astype(int)

	tile_stats = (
		poly_df.groupby(["tile_x", "tile_y"], as_index=False)
		.agg(n_cells=("cell_id", "count"), n_known=("known", "sum"))
		.assign(
			known_fraction=lambda d: d["n_known"] / d["n_cells"],
			score=lambda d: (d["known_fraction"] ** 2) * np.log1p(d["n_cells"]),
		)
	)

	selected = _weighted_tile_sample(tile_stats, n_crops=n_crops, seed=seed)
	selected = selected.sort_values(["score", "known_fraction", "n_cells"], ascending=[False, False, False]).reset_index(drop=True)

	tif_path = str(morph_path)
	with tifffile.TiffFile(tif_path) as tif:
		series = tif.series[0]
		axes = series.axes

		try:
			import zarr

			arr = _resolve_zarr_array(zarr.open(series.aszarr(), mode="r"))
			use_zarr = True
		except Exception as exc:
			warnings.warn(f"zarr path unavailable ({exc}); falling back to tifffile reads")
			arr = None
			use_zarr = False

		max_h = int(series.shape[axes.find("Y")])
		max_w = int(series.shape[axes.find("X")])

		written = []
		for row in selected.itertuples(index=False):
			tx, ty = int(row.tile_x), int(row.tile_y)
			x0 = tx * n_pixel
			y0 = ty * n_pixel
			x1 = min(x0 + n_pixel, max_w)
			y1 = min(y0 + n_pixel, max_h)

			if x0 >= max_w or y0 >= max_h or x1 <= x0 or y1 <= y0:
				continue

			if use_zarr:
				img = _read_tile_mip(series, axes, (x0, y0, x1, y1), arr=arr)
			else:
				img = _read_tile_mip(series, axes, (x0, y0, x1, y1), arr=None)

			overlap = poly_df[
				(poly_df["maxx"] > x0)
				& (poly_df["minx"] < x1)
				& (poly_df["maxy"] > y0)
				& (poly_df["miny"] < y1)
			]
			mask = _draw_tile_masks(overlap[["polygon", "group_id"]], x0=x0, y0=y0, x1=x1, y1=y1)

			if img.shape != (n_pixel, n_pixel):
				img = _pad_to_square(img, n_pixel=n_pixel)
			if mask.shape != (n_pixel, n_pixel):
				mask = _pad_to_square(mask, n_pixel=n_pixel)

			img_name = f"{root}_{tx}_{ty}.tif"
			mask_name = f"{root}_{tx}_{ty}_masks.tif"
			img_path = outdir / img_name
			mask_path = outdir / mask_name

			tifffile.imwrite(str(img_path), img)
			tifffile.imwrite(str(mask_path), mask)

			written.append(
				{
					"x": tx,
					"y": ty,
					"image": str(img_path),
					"mask": str(mask_path),
					"n_cells": int(row.n_cells),
					"known_fraction": float(row.known_fraction),
				}
			)

	if not written:
		raise RuntimeError("No crops were written. Check n_pixel and Xenium image dimensions.")

	summary = pd.DataFrame(written).sort_values(["known_fraction", "n_cells"], ascending=[False, False])
	summary_path = outdir / f"{root}_selected_crops.tsv"
	summary.to_csv(summary_path, sep="\t", index=False)

	print(f"Wrote {len(summary)} crops to {outdir}")
	print(f"Group mapping: {map_path}")
	print(f"Crop summary: {summary_path}")
	return summary
