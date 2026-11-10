"""Glass-brain plot of the most relevant ROIs of the trained FP-GCN layer.

Input : CSV exported with HadamardGCNConv.to_csv (rows = ROIs, cols = metrics)
Output: PNG with the top-k ROIs drawn as markers whose color and size are
        proportional to the mean |W| of the ROI.

Usage
-----
python plot_glass_brain_top_rois.py weights.csv --top-k 10 --out top_rois.png
python plot_glass_brain_top_rois.py weights.csv --modality RS      # only RS-* columns
python plot_glass_brain_top_rois.py weights.csv --coords my_centroids.csv

Requires: nilearn, matplotlib, pandas, numpy
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import plotting

# --------------------------------------------------------------------------
# APPROXIMATE MNI centroids (x, y, z in mm) of the Desikan-Killiany / aseg
# ROIs most likely to appear in the top ranking. They are representative
# values for visualization. For publication, replace them with centroids
# computed from YOUR atlas (see `centroids_from_atlas` below) and pass them
# with --coords.
# --------------------------------------------------------------------------
APPROX_MNI = {
    'ctx.rh.precuneus':          (9, -58, 40),
    'ctx.lh.precuneus':          (-9, -58, 40),
    'ctx.rh.superiorparietal':   (24, -60, 54),
    'ctx.lh.superiorparietal':   (-24, -60, 54),
    'ctx.rh.isthmuscingulate':   (8, -48, 12),
    'ctx.lh.isthmuscingulate':   (-8, -48, 12),
    'ctx.rh.lateraloccipital':   (32, -86, 4),
    'ctx.lh.lateraloccipital':   (-32, -86, 4),
    'ctx.lh.precentral':         (-38, -10, 42),
    'ctx.rh.precentral':         (38, -10, 42),
    'ctx.lh.inferiorparietal':   (-42, -66, 30),
    'ctx.rh.inferiorparietal':   (42, -66, 30),
    'ctx.lh.middletemporal':     (-58, -30, -8),
    'ctx.rh.middletemporal':     (58, -30, -8),
    'ctx.lh.superiorfrontal':    (-14, 36, 44),
    'ctx.rh.superiorfrontal':    (14, 36, 44),
    'ctx.lh.cuneus':             (-8, -80, 24),
    'ctx.rh.cuneus':             (8, -80, 24),
    'left.thalamus':             (-11, -18, 8),
    'right.thalamus':            (11, -18, 8),
    'left.pallidum':             (-20, -4, 0),
    'right.pallidum':            (20, -4, 0),
    'left.hippocampus':          (-26, -22, -14),
    'right.hippocampus':         (26, -22, -14),
}


def load_coords(path=None):
    """Return {roi: (x, y, z)}; user CSV (columns: ROI,x,y,z) overrides."""
    coords = dict(APPROX_MNI)
    if path is not None:
        df = pd.read_csv(path, index_col=0)
        for roi, row in df.iterrows():
            coords[roi] = (row['x'], row['y'], row['z'])
    return coords


def centroids_from_atlas(labels_img, label_to_roi):
    """Optional helper: exact centroids from an atlas in MNI space.

    labels_img   : path/NIfTI with one integer label per ROI
    label_to_roi : {label_value: roi_name} using the same names as the CSV
    Returns a DataFrame (index=ROI, columns=x,y,z) that you can save and
    pass with --coords.
    """
    coords, labels = plotting.find_parcellation_cut_coords(labels_img, return_label_names=True)
    rows = {label_to_roi[int(l)]: c for l, c in zip(labels, coords)
            if int(l) in label_to_roi}
    return pd.DataFrame.from_dict(rows, orient='index', columns=['x', 'y', 'z'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv')
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--modality', choices=['FA', 'GM', 'RS'], default=None,
                   help='restrict importance to one modality (columns FA-*, ...)')
    p.add_argument('--coords', default=None,
                   help='CSV with columns ROI,x,y,z overriding the defaults')
    p.add_argument('--mode', default='ortho',
                   help="nilearn display_mode, e.g. 'ortho', 'lyrz', 'xz'")
    p.add_argument('--out', default='top_rois_glass_brain.png')
    args = p.parse_args()

    W = pd.read_csv(args.csv, index_col=0).abs()
    if args.modality:
        W = W[[c for c in W.columns if c.startswith(args.modality + '-')]]

    roi_importance = W.mean(axis=1).sort_values(ascending=False)
    top = roi_importance.head(args.top_k)

    coords = load_coords(args.coords)
    missing = [r for r in top.index if r not in coords]
    if missing:
        raise KeyError(f'No coordinates for {missing}. Add them with --coords '
                       f'(CSV: ROI,x,y,z).')

    node_coords = np.array([coords[r] for r in top.index], dtype=float)
    values = top.values

    # marker size grows with importance
    sizes = 40 + 360 * (values - values.min()) / (np.ptp(values) + 1e-12)

    display = plotting.plot_markers(
        node_values=values,
        node_coords=node_coords,
        node_size=sizes,
        node_cmap=plt.get_cmap('YlOrRd'),
        node_vmin=0.0,
        node_vmax=float(values.max()),
        display_mode=args.mode,
        colorbar=True,
        alpha=0.9,
        #title=f'Top-{args.top_k} ROIs by mean |W|' (f' ({args.modality})' if args.modality else ''),
    )
    display.savefig(args.out, dpi=300)
    display.close()

    print(top.round(3).to_string())
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
