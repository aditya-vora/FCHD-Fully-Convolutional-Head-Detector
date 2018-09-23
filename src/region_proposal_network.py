import numpy as np
from torch.nn import functional as F
import torch as t
from torch import nn

from src.bbox_tools import generate_anchor_base
from src.creator_tool import ProposalCreator


class RegionProposalNetwork(nn.Module):
    """ Generate anchors and classify the anchors. Compute two heads
        1) Regression head: Compute the shift and scale of the anchor to
            accurately localize the head. 
        2) Classification head: Compute the probability that the anchor 
            contains the head. 

    Args: 
        in_channels: number of input channels to the convolutional layer
        mid_channels: number of convolutional filters. 
        ratios: what ratios of the anchors are required i.e. width and height
        anchor_scales: scales of the anchors. 
        proposal_creator_params: Current weights of the network.
    """
    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(
            anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        """ Forward pass function to the network. 

        Args: 
            x : feature size
            img_size: size of the image.

        Returns: 
            rpn_locs: the scales and translates of the anchors.
            rpn_scores: probability score of the anchors 
            rois: mapped region proposal from the scales and the translates. 
            rois_scores: scores of the rois. 
            anchors: anchors that are used to compute the proposals.
        """
        n, _, hh, ww = x.size()
        # Generate anchors throughout the image.
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)
        n_anchor = anchor.shape[0] // (hh * ww)
        h = F.relu(self.conv1(x))                               # (1,512,30,40)
        rpn_locs = self.loc(h)                                  # (1,12,30,40)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)     # (1,3600,4)
        rpn_scores = self.score(h)                              # (1,6,30,40)
        
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_fg_scores = \
            rpn_scores.view(n, hh, ww, n_anchor, 2)[:, :, :, :, 1].contiguous() # (1,30,40,3)
        rpn_fg_scores = rpn_fg_scores.view(n, -1)           # (1,3600)
        rpn_scores = rpn_scores.view(n, -1, 2)              # (1,3600,2)


        # Map the scales and translates to the rois.
        rois, rois_scores = self.proposal_layer(
            rpn_locs[0].cpu().data.numpy(),
            rpn_fg_scores[0].cpu().data.numpy(),
            anchor, img_size, 
            scale=scale
        )

        return rpn_locs, rpn_scores, rois, rois_scores, anchor

def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):    
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
