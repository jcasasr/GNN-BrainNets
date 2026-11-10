from typing import Sequence, Tuple
import logging
import numpy as np
from sklearn import preprocessing as p
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import Optional, Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.inits import ones, zeros
from torch_geometric.typing import Adj, OptTensor
from gnns.FPGCN import FPGCN


METRICS_NAME = ["degree", "strength", "betweenness", "closeness", "local_eff", "clustering", "pagerank", "eigenvector"]
MATRICES_EMBBEDINGS = ["FA", "GM", "RS"]
# ROIs
rois = ['ctx.lh.caudalanteriorcingulate', 'ctx.lh.caudalmiddlefrontal', 'ctx.lh.cuneus', 'ctx.lh.entorhinal', 'ctx.lh.fusiform', 'ctx.lh.inferiorparietal',
        'ctx.lh.inferiortemporal', 'ctx.lh.isthmuscingulate', 'ctx.lh.lateraloccipital', 'ctx.lh.lateralorbitofrontal', 'ctx.lh.lingual', 'ctx.lh.medialorbitofrontal', 
        'ctx.lh.middletemporal', 'ctx.lh.parahippocampal', 'ctx.lh.paracentral', 'ctx.lh.parsopercularis', 'ctx.lh.parsorbitalis', 'ctx.lh.parstriangularis',
        'ctx.lh.pericalcarine', 'ctx.lh.postcentral', 'ctx.lh.posteriorcingulate', 'ctx.lh.precentral', 'ctx.lh.precuneus', 'ctx.lh.rostralanteriorcingulate', 
        'ctx.lh.rostralmiddlefrontal', 'ctx.lh.superiorfrontal', 'ctx.lh.superiorparietal', 'ctx.lh.superiortemporal', 'ctx.lh.supramarginal', 'ctx.lh.transversetemporal',
        'ctx.lh.insula', 'left.thalamus', 'left.caudate', 'left.putamen', 'left.pallidum', 'left.hippocampus', 'left.amygdala', 'left.accumbens', 'right.thalamus', 
        'right.caudate', 'right.putamen', 'right.pallidum', 'right.hippocampus', 'right.amygdala', 'right.accumbens', 'ctx.rh.caudalanteriorcingulate', 'ctx.rh.caudalmiddlefrontal', 
        'ctx.rh.cuneus', 'ctx.rh.entorhinal', 'ctx.rh.fusiform', 'ctx.rh.inferiorparietal', 'ctx.rh.inferiortemporal', 'ctx.rh.isthmuscingulate', 'ctx.rh.lateraloccipital',
        'ctx.rh.lateralorbitofrontal', 'ctx.rh.lingual', 'ctx.rh.medialorbitofrontal', 'ctx.rh.middletemporal', 'ctx.rh.parahippocampal', 'ctx.rh.paracentral',
        'ctx.rh.parsopercularis', 'ctx.rh.parsorbitalis', 'ctx.rh.parstriangularis', 'ctx.rh.pericalcarine', 'ctx.rh.postcentral', 'ctx.rh.posteriorcingulate', 'ctx.rh.precentral', 
        'ctx.rh.precuneus', 'ctx.rh.rostralanteriorcingulate', 'ctx.rh.rostralmiddlefrontal', 'ctx.rh.superiorfrontal', 'ctx.rh.superiorparietal', 'ctx.rh.superiortemporal',
        'ctx.rh.supramarginal', 'ctx.rh.transversetemporal', 'ctx.rh.insula']
# Export the weight matrix as a CSV and as a Headmap
metrics = [matrix+"-"+metric for matrix in MATRICES_EMBBEDINGS for metric in METRICS_NAME]


def get_W_as_PD(W:np.ndarray, roi_names: Optional[Sequence[str]] = None,
                        feature_names: Optional[Sequence[str]] = None,
                        absolute: bool = False,
                        MinMaxScaler: bool = False) -> pd.DataFrame:
    r"""Returns the learned Hadamard weight matrix :math:`W` as a
    :class:`pandas.DataFrame` of shape :obj:`[num_nodes, channels]`
    (rows: ROIs, columns: metrics).

    Args:
        roi_names: Names of the :obj:`num_nodes` ROIs (row order).
        feature_names: Names of the :obj:`channels` metrics (column order).
        absolute: If :obj:`True`, returns :math:`|W|`.
    """
    if absolute:
        W = np.abs(W)

    if MinMaxScaler:
        min_max_scaler = p.MinMaxScaler()
        W = min_max_scaler.fit_transform(W)
    
    return pd.DataFrame(W, index=roi_names, columns=feature_names)


def get_importance(W: np.ndarray, roi_names: Optional[Sequence[str]] = None,
                    feature_names: Optional[Sequence[str]] = None):
    r"""Returns ``(feature_importance, roi_importance)`` as pandas Series:
    :math:`s_j = \frac{1}{N}\sum_i |W_{ij}|` and
    :math:`r_i = \frac{1}{F}\sum_j |W_{ij}|`, sorted in descending order.
    """
    absW = get_W_as_PD(W, roi_names, feature_names, absolute=True)
    return (absW.mean(axis=0).sort_values(ascending=False),
            absW.mean(axis=1).sort_values(ascending=False))

def to_csv(df: pd.DataFrame, path: str, roi_names: Optional[Sequence[str]] = None,
            feature_names: Optional[Sequence[str]] = None,
            absolute: bool = False, **kwargs) -> None:
    r"""Exports the weight matrix (rows: ROIs, columns: metrics) to CSV.
    Extra keyword arguments are forwarded to
    :meth:`pandas.DataFrame.to_csv`."""
    df.to_csv(path, index_label='ROI', **kwargs)


def plot_heatmap(
    df: 'pd.DataFrame',
    roi_names: Optional[Sequence[str]] = None,
    feature_names: Optional[Sequence[str]] = None,
    absolute: bool = False,
    sort_rois: bool = False,
    annotate: Optional[bool] = None,
    cmap: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
    ax=None,
):
    r"""Plots the weight matrix as a heatmap (ROIs x metrics).

    Args:
        roi_names, feature_names: Axis labels.
        absolute: Plot :math:`|W|` (sequential colormap) instead of the
            signed :math:`W` (diverging colormap centered at 0).
        sort_rois: Sort ROIs by mean :math:`|W|` (most relevant on top).
        annotate: Write the values in the cells. Defaults to :obj:`True`
            only for small matrices (<= 200 cells).
        cmap: Matplotlib colormap. Defaults to ``'magma'`` if
            :obj:`absolute` else ``'RdBu_r'``.
        figsize: Figure size. Chosen automatically if :obj:`None`.
        save_path: If given, the figure is saved there.
        ax: Optional existing matplotlib axes to draw on.

    Returns:
        ``(fig, ax)``
    """
    if sort_rois:
        order = df.abs().mean(axis=1).sort_values(ascending=False).index
        df = df.loc[order]

    n_rois, n_feats = df.shape
    vals = df.values

    # Possible cmaps: 'magma', 'RdBu_r'
    if absolute:
        cmap = cmap or 'Reds'
        vmin, vmax = 0.0, float(vals.max())
    else:
        cmap = cmap or 'Reds'
        vmax = float(np.abs(vals).max())
        vmin = float(np.abs(vals).min()) #-vmax  # symmetric so that 0 is the neutral color

    if ax is None:
        figsize = figsize or (max(6.0, 0.4 * n_feats + 3.0), max(4.0, 0.18 * n_rois + 2.0))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(vals, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label='|W|' if absolute else 'W')

    ax.set_xticks(range(n_feats))
    ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=12)
    ax.set_yticks(range(n_rois))
    ax.set_yticklabels(df.index, fontsize=12)
    #ax.set_xlabel('Graph metric', fontsize=12)
    #ax.set_ylabel('ROI', fontsize=12)
    #ax.set_title(title or ('|W|' if absolute else 'W') +' (Hadamard weights)')

    if annotate is None:
        annotate = vals.size <= 200
    if annotate:
        for i in range(n_rois):
            for j in range(n_feats):
                if absolute:  # magma: dark for low values
                    light_text = vals[i, j] < 0.5 * vmax
                else:  # RdBu_r: dark at both extremes
                    light_text = abs(vals[i, j]) > 0.6 * vmax
                ax.text(j, i, f'{vals[i, j]:.2f}', ha='center',
                        va='center', fontsize=6,
                        color='white' if light_text else 'black')

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig, ax


MODEL_NAMES = ["Step-7-1_MM_FPGCN-0-0_lr0.0001_wd0.0005_noDA", 
               "Step-7-1_MM-THR_FPGCN-0-0_lr0.0001_wd0.0005_noDA"]

for model_name in MODEL_NAMES:
    print("\n*** Processing model: {}\n".format(model_name))
    # Load model
    model = FPGCN(dim_in=len(metrics), dim_out=2, logger=logging.getLogger(__name__))
    model.load_state_dict(torch.load("./models/"+ model_name +".pt", map_location="cpu", weights_only=False))

    # get the weight matrix as a Numpy matrix and Pandas DataFrame
    conv = model.conv1
    W = conv.get_weight_matrix()
    df = get_W_as_PD(W=W, roi_names=rois, feature_names=metrics, absolute=False, MinMaxScaler=True)

    to_csv(df=df, path="./results/"+ model_name +"_weights.csv", roi_names=rois, feature_names=metrics)
    plot_heatmap(df=df, roi_names=rois, feature_names=metrics, absolute=True, sort_rois=False, save_path="./results/"+ model_name +"_heatmap_abs.png")
    plot_heatmap(df=df, roi_names=rois, feature_names=metrics, absolute=False, sort_rois=False, save_path="./results/"+ model_name +"_heatmap_signed.png")

    feat_imp, roi_imp = get_importance(W=W, roi_names=rois, feature_names=metrics)
    print("\n*** Feature importance:")
    print(feat_imp)              # global metric ranking
    print("\n*** ROI importance:")
    print(roi_imp.head(10))      # top-10 ROIs
