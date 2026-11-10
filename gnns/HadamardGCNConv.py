from typing import Sequence, Tuple
import numpy as np
from sklearn import preprocessing as p

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.inits import ones, zeros
from torch_geometric.typing import Adj, OptTensor


class _Identity(torch.nn.Identity):
    """`torch.nn.Identity` that also answers to `reset_parameters()`."""
    def reset_parameters(self) -> None:
        pass


class HadamardGCNConv1D(GCNConv):
    """
    GCNConv in which the feature transformation is an element-wise
    (Hadamard) product instead of a matrix product:

    .. math::
        \mathbf{X}^{\prime} = \left( \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \right) \odot \mathbf{w}
        + \mathbf{b}

    with :math:`\mathbf{w} \in \mathbb{R}^{F}` broadcast over the node
    dimension, i.e. a diagonal weight matrix.  Hence
    ``out_channels == in_channels``.
    """
    def __init__(self, channels: int, bias: bool = True, **kwargs):
        # Build a regular GCNConv without its bias, then disable its linear
        # transform so that `GCNConv.forward` returns exactly `A @ X`.
        super().__init__(channels, channels, bias=False, **kwargs)
        self.lin = _Identity()

        self.weight = Parameter(torch.empty(channels))
        if bias:
            self.elem_bias = Parameter(torch.empty(channels))
        else:
            self.register_parameter('elem_bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if hasattr(self, 'weight'):  # Skip the call made inside `__init__`.
            ones(self.weight)
            zeros(self.elem_bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        out = super().forward(x, edge_index, edge_weight) * self.weight
        if self.elem_bias is not None:
            out = out + self.elem_bias
        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels})'


class HadamardGCNConv2D(GCNConv):
    r"""GCNConv where the feature transformation is a Hadamard product with a
    weight matrix of shape :obj:`[num_nodes, channels]`:

    .. math::
        \mathbf{X}^{\prime} = \left( \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \right) \odot \mathbf{W}
        + \mathbf{b}

    All graphs must have exactly :obj:`num_nodes` nodes.  For mini-batches
    (nodes of consecutive graphs concatenated, as produced by PyG's
    :class:`~torch_geometric.loader.DataLoader`), the same :math:`\mathbf{W}`
    is applied to every graph in the batch.

    Args:
        num_nodes (int): Number of nodes per graph.
        channels (int): Number of input/output features (they must match for
            a Hadamard product).
        bias (bool): Add a learnable per-feature bias. (default: :obj:`True`)
        **kwargs: Forwarded to :class:`GCNConv` (`improved`, `cached`,
            `add_self_loops`, `normalize`, ...).
    """
    def __init__(self, num_nodes: int, channels: int, bias: bool = True, **kwargs):
        # bias=False: the parent must return exactly `A @ X`.
        super().__init__(channels, channels, bias=False, **kwargs)
        self.lin = _Identity()  # disable the parent's linear transform

        self.num_nodes = num_nodes
        self.weight = Parameter(torch.empty(num_nodes, channels))
        if bias:
            self.elem_bias = Parameter(torch.empty(channels))
        else:
            self.register_parameter('elem_bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if hasattr(self, 'weight'):  # skip the call made inside `__init__`
            ones(self.weight)
            zeros(self.elem_bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        if x.size(0) % self.num_nodes != 0:
            raise ValueError(
                f"Number of nodes ({x.size(0)}) is not a multiple of "
                f"num_nodes={self.num_nodes}; all graphs must have exactly "
                f"{self.num_nodes} nodes.")

        out = super().forward(x, edge_index, edge_weight)  # A @ X

        # [B*N, F] -> [B, N, F]; W [N, F] broadcasts over the batch dim.
        out = out.view(-1, self.num_nodes, self.in_channels) * self.weight
        out = out.reshape(-1, self.in_channels)

        if self.elem_bias is not None:
            out = out + self.elem_bias
        return out
 
    @torch.no_grad()
    def get_weight_matrix(self):
        r"""Returns the learned Hadamard weight matrix :math:`W` as a
        :class:`np.ndarray` of shape :obj:`[num_nodes, channels]`
        """
        W = self.weight.detach().cpu().numpy()
        return W
 
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_nodes={self.num_nodes}, '
                f'channels={self.in_channels})')