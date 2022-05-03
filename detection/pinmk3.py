from collections import OrderedDict
from typing import Dict

import torch

from detection.pinmk2 import ParticleIdentificationNetworkMK2
from detection.utils import FocalLoss


class ParticleIdentificationNetworkMK3(torch.nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.

    The feature maps are currently supposed to be in increasing depth
    order.

    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.

    Args:
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names

    Examples::

        >>> m = torchvision.ops.ParticleIdentificationNetworkMK3([10, 20, 30], 5)
        >>> # get some dummy data
        >>> x = OrderedDict()
        >>> x['feat0'] = torch.rand(1, 10, 64, 64)
        >>> x['feat2'] = torch.rand(1, 20, 16, 16)
        >>> x['feat3'] = torch.rand(1, 30, 8, 8)
        >>> # compute the FPN on top of x
        >>> output = m(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('feat0', torch.Size([1, 5, 64, 64])),
        >>>    ('feat2', torch.Size([1, 5, 16, 16])),
        >>>    ('feat3', torch.Size([1, 5, 8, 8]))]

    """

    def __init__(
        self,
        in_channels_list=[512, 1024, 2048],
        out_channels=1,
    ):
        super(ParticleIdentificationNetworkMK3, self).__init__()
        self.inner_blocks = torch.nn.ModuleList()
        self.layer_blocks = torch.nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = torch.nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = torch.nn.Conv2d(
                out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

        self.mk2 = ParticleIdentificationNetworkMK2()

    def get_result_from_inner_blocks(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x):
        feat_maps = self.mk2.compute_feat_map_for_mk3(x)
        fpn_maps = self.compute_fpn_outputs(feat_maps)
        heatmaps = OrderedDict()
        for name in fpn_maps:
            x = fpn_maps[name]
            x = x.permute([0, 2, 3, 1])
            x = torch.squeeze(x, dim=-1)
            x = torch.sigmoid(x)
            heatmaps[name] = x
        return heatmaps

    def make_prediction(self, x):
        feat_maps = self.mk2.compute_feat_map_for_mk3(x)
        fpn_maps = self.compute_fpn_outputs(feat_maps)
        agg_heatmap = []
        for name in fpn_maps:
            x = fpn_maps[name]
            x = x.permute([0, 2, 3, 1])
            x = torch.squeeze(x, dim=-1)
            x = torch.sigmoid(x)
            agg_heatmap.append(x)
        agg_heatmap = torch.max(torch.stack(agg_heatmap, dim=0), dim=0)[
            0].reshape([-1])
        return agg_heatmap

    def compute_fpn_outputs(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Computes the FPN for a set of feature maps.

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = torch.nn.functional.interpolate(
                last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(
                0, self.get_result_from_layer_blocks(last_inner, idx))

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out

    def compute_loss(self, input, rank):
        batch_size = input["gt_heatmap"].shape[0]
        loss = FocalLoss(alpha=.65, gamma=2.0, reduction="sum")
        heatmaps = self.forward(input["img"].cuda(rank))
        gt_heatmap = input["gt_heatmap"].cuda(rank)
        # feat2_loss = loss(heatmaps["feat2"], gt_heatmap) / batch_size
        # feat3_loss = loss(heatmaps["feat3"], gt_heatmap) / batch_size
        # feat4_loss = loss(heatmaps["feat4"], gt_heatmap) / batch_size
        feat2_loss = torch.nn.functional.binary_cross_entropy(
            heatmaps["feat2"], gt_heatmap, reduction="sum") / batch_size
        feat3_loss = torch.nn.functional.binary_cross_entropy(
            heatmaps["feat3"], gt_heatmap, reduction="sum") / batch_size
        feat4_loss = torch.nn.functional.binary_cross_entropy(
            heatmaps["feat4"], gt_heatmap, reduction="sum") / batch_size
        return {"feat2_loss": feat2_loss, "feat3_loss": feat3_loss, "feat4_loss": feat4_loss}
