from __future__ import division

import os
import numpy as np 
from torch.autograd import Variable
from torch.utils import data as data_
import torch
import random
import ipdb

from src.head_detector_vgg16 import Head_Detector_VGG16
from trainer import Head_Detector_Trainer
from src.config import opt
import src.utils as utils 
from data.dataset import Dataset, inverse_normalize
import src.array_tool as at
from src.vis_tool import visdom_bbox
from src.bbox_tools import bbox_iou

phases = ['train', 'val', 'test']

data_check_flag = False

def eval(dataloader, head_detector):
    """
    Given the dataloader of the test split compute the
    average corLoc of the dataset using the head detector 
    model given as the argument to the function. 
    """
    test_img_num = 0
    test_corrLoc = 0.0
    for _, (img, bbox_, scale) in enumerate(dataloader):
        scale = at.scalar(scale)
        img, bbox = img.cuda().float(), bbox_.cuda()
        img, bbox = Variable(img), Variable(bbox)
        pred_bboxes_, _ = head_detector.predict(img, scale, mode='evaluate')
        gt_bboxs = at.tonumpy(bbox_)[0]
        pred_bboxes_ = at.tonumpy(pred_bboxes_)
        if pred_bboxes_.shape[0] == 0:
            test_img_num += 1
            continue
        else:
            ious = bbox_iou(pred_bboxes_, gt_bboxs)
            max_ious = ious.max(axis=1)
            corr_preds = np.where(max_ious >= 0.5)[0]
            num_boxs = gt_bboxs.shape[0]
            num_corr_preds = len(corr_preds)
            test_corrLoc += num_corr_preds / num_boxs
            test_img_num += 1
    return test_corrLoc / test_img_num


def train():    
    # Get the dataset
    for phase in phases:
        if phase == 'train':
            train_data_list_path = os.path.join(opt.data_root_path,'brainwash_train.idl')
            train_data_list = utils.get_phase_data_list(train_data_list_path)
        elif phase == 'val':
            val_data_list_path = os.path.join(opt.data_root_path,'brainwash_val.idl')
            val_data_list = utils.get_phase_data_list(val_data_list_path)
        elif phase == 'test':
            test_data_list_path = os.path.join(opt.data_root_path,'brainwash_test.idl')
            test_data_list = utils.get_phase_data_list(test_data_list_path)
    
    print "Number of images for training: %s" %(len(train_data_list))
    print "Number of images for val: %s" %(len(val_data_list))
    print "Number of images for test: %s" %(len(test_data_list))


    # Just to visualize if the dataset is correctly loaded
    if data_check_flag: 
        utils.check_loaded_data(train_data_list[random.randint(1,len(train_data_list))])
        utils.check_loaded_data(val_data_list[random.randint(1,len(val_data_list))])
        utils.check_loaded_data(test_data_list[random.randint(1,len(test_data_list))])

    # Load the train dataset
    dataset = Dataset(train_data_list)
    
    # Load the test dataset
    test_dataset = Dataset(val_data_list)
    print "Load data." 
    
    dataloader = data_.DataLoader(dataset, batch_size=1,shuffle=True, num_workers=1)
    test_dataloader = data_.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

    # Initialize the head detector.
    head_detector_vgg16 = Head_Detector_VGG16(ratios=[1], anchor_scales=[8,16])
    print('model construct completed')
    
    trainer = Head_Detector_Trainer(head_detector_vgg16).cuda()
    best_map = 0.0 
    lr_ = opt.lr

    for epoch in range(opt.epoch):
        trainer.reset_meters()
        # train_img_num = 0
        # train_corrLoc = 0.0 
        for ii, (img, bbox_, scale) in enumerate(dataloader):
            scale = at.scalar(scale)
            img, bbox = img.cuda().float(), bbox_.cuda()
            img, bbox = Variable(img), Variable(bbox)
            _, _, _ = trainer.train_step(img, bbox, scale)
            
            if (ii+1) % opt.plot_every == 0:
                # if os.path.exists(opt.debug_file):
                #     ipdb.set_trace()
                trainer.vis.plot_many(trainer.get_meter_data())
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_, at.tonumpy(bbox_[0]))
                trainer.vis.img('gt_img', gt_img)
                rois,_ = trainer.head_detector.predict(img, scale=scale, mode='visualize')
                pred_img = visdom_bbox(ori_img_, at.tonumpy(rois))
                trainer.vis.img('pred_img', pred_img)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
        
        # avg_train_CorrLoc = train_corrLoc / train_img_num

        avg_test_CorrLoc = eval(test_dataloader, head_detector_vgg16)
        
        print("Epoch {} of {}.".format(epoch+1, opt.epoch))
        # print("  training average corrLoc accuracy:\t\t{:.3f}".format(avg_train_CorrLoc))
        print("  test average corrLoc accuracy:\t\t{:.3f}".format(avg_test_CorrLoc))
		
        model_save_path = trainer.save(best_map=avg_test_CorrLoc)
		
        # if avg_test_CorrLoc >= best_map:
        #     best_map = avg_test_CorrLoc
        #     best_path = trainer.save(best_map=best_map)
        if epoch == 8:
            trainer.load(model_save_path)
            trainer.head_detector.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay


if __name__ == '__main__':

    train()
