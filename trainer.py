from __future__ import print_function

from collections import namedtuple
import time
from torch.nn import functional as F
from torch import nn
import torch as t
from torch.autograd import Variable
from torchnet.meter import ConfusionMeter, AverageValueMeter
import os
from src.creator_tool import AnchorTargetCreator
import src.array_tool as at
from src.vis_tool import Visualizer
from src.config import opt

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'total_loss'
                        ])

class Head_Detector_Trainer(nn.Module):
    def __init__(self, head_detector):
        super(Head_Detector_Trainer, self).__init__()
        self.head_detector = head_detector
        self.rpn_sigma = opt.rpn_sigma
        self.anchor_target_creator = AnchorTargetCreator()
        self.optimizer = self.head_detector.get_optimizer()
        self.vis = Visualizer(env=opt.env)
        self.rpn_cm = ConfusionMeter(2)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def forward(self, imgs, bboxs, scale):
        n,_,_ = bboxs.size()
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')        
        _, _, H, W = imgs.size()
        img_size = (H, W)
        features = self.head_detector.extractor(imgs)
        rpn_locs, rpn_scores, rois, rois_scores, anchor = self.head_detector.rpn(features, img_size, scale)
        bbox = bboxs[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(at.tonumpy(bbox),anchor,img_size)
        gt_rpn_label = at.tovariable(gt_rpn_label).long()
        gt_rpn_loc = at.tovariable(gt_rpn_loc)
        rpn_loc_loss = head_detector_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())
        losses = [rpn_loc_loss, rpn_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses), rois, rois_scores

    def train_step(self, imgs, bboxes, scale):
        self.optimizer.zero_grad()
        losses, rois, rois_scores = self.forward(imgs, bboxes, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses, rois, rois_scores

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        save_dict = dict()
        save_dict['model'] = self.head_detector.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = os.path.join(opt.model_save_path, 'head_detector%s' % timestr)             
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        t.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.head_detector.load_state_dict(state_dict['model'])
        else:  
            self.head_detector.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses):
        loss_d = {k: at.scalar(at.tonumpy(v)) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}

def _smooth_l1_loss(x, t, in_weight, sigma):
    t = t.float()
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    flag = Variable(flag)
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()

def head_detector_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.size()).cuda()
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, Variable(in_weight), sigma)
    loc_loss /= (gt_label >= 0).sum().float()  # ignore gt_label==-1 for rpn_loss
    return loc_loss
