from __future__ import division
import torch as t
import numpy as np
import cupy as cp
from src import array_tool as at
from src.bbox_tools import loc2bbox
from src.nms import non_maximum_suppression

from torch import nn
from data.dataset import preprocess
from torch.nn import functional as F
from src.config import opt

class Head_Detector(nn.Module):
    def __init__(self, extractor, rpn):
        super(Head_Detector, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        # self.nms_thresh = 0.3 
        # self.score_thresh = 0.005

    def forward(self, x, scale=1.):
        _, _, H, W = x.size()
        img_size = (H, W)
        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, rois_scores, anchor = self.rpn(h, img_size, scale)
        return rpn_locs, rpn_scores, rois, rois_scores, anchor
    
    def get_optimizer(self):
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = t.optim.Adam(params)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer


    def _suppress(self, raw_cls_bbox, raw_prob, nms_thresh, score_thresh):
        # bbox = list()
        # score = list()
        mask = raw_prob > score_thresh
        bbox = raw_cls_bbox[mask]
        scores = raw_prob[mask]
        keep = non_maximum_suppression(
            cp.array(bbox), nms_thresh, scores)
        keep = cp.asnumpy(keep)
        bbox = bbox[keep]
        scores = scores[keep]
#         bbox.append(cls_bbox_l[keep])
#         score.append(prob_l[keep])
#         bbox = np.concatenate(bbox, axis=0).astype(np.float32)
#         score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, scores


    def predict(self, x, scale=1., mode='evaluate', thresh=0.01):
        
        if mode == 'evaluate':
            nms_thresh = 0.3
            score_thresh = thresh
        elif mode == 'visualize':
            nms_thresh = 0.3 
            score_thresh = thresh

        _, _, rois, rois_scores, _ = self.forward(x, scale=scale)
        roi = at.totensor(rois)
        probabilities = at.tonumpy(F.softmax(at.tovariable(rois_scores)))
        _, _, H, W = x.size()
        size = (H,W)
        roi[:, 0::2] = (roi[:, 0::2]).clamp(min=0, max=size[0])
        roi[:, 1::2] = (roi[:, 1::2]).clamp(min=0, max=size[1])        
        roi_raw = at.tonumpy(roi)
        probabilities = np.squeeze(probabilities)
        bbox, score = self._suppress(roi_raw, probabilities, nms_thresh, score_thresh)
        return bbox, score

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer

    