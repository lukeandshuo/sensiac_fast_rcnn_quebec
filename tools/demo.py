#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg,get_output_dir
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
from datasets.factory import get_imdb
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
CLASSES = ('__background__',
           'vehicle')

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}


def vis_detections(im, class_name, bbox,score, gt,thresh=0.5):
    """Draw detected bounding boxes."""
   # inds = np.where(dets[:, -1] >= thresh)[0]
    cv2.putText(im,'{:s} {:.3f}'.format(class_name,score),(bbox[0],bbox[1]-3),0,0.6,(0,0,255))
    cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0,0),2)
    cv2.rectangle(im, (gt[0], gt[1]), (gt[2], gt[3]), (0, 255, 0), 2)
    cv2.imshow("result",im)
    cv2.waitKey(1)

def IOU(bb,gt):
    ixmin = np.maximum(gt[0], bb[0])
    iymin = np.maximum(gt[1], bb[1])
    ixmax = np.minimum(gt[2], bb[2])
    iymax = np.minimum(gt[3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih
    # union
    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
            (gt[2] - gt[0] + 1.) *
            (gt[3] - gt[1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps    

def demo(net,classes):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load pre-computed Selected Search object proposals
    # box_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo',
    #                         image_name + '_boxes.mat')
    # obj_proposals = sio.loadmat(box_file)['boxes']
    #
    # # Load the demo image
    # im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name + '.jpg')
    # im = cv2.imread(im_file)
    miss_num =0
    imdb = get_imdb('sensiac_test')
    num_images = len(imdb.image_index)
    roidb = imdb.roidb
    GT=imdb.gt_roidb()
    print "size of images:",num_images
    missed_frame = []
    missed_frame_ind = []
    for i in xrange(num_images):
        if i > 100: #test
            break
        im = cv2.imread(imdb.image_path_at(i))

    # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(net, im, roidb[i]['boxes'])
        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0])

        # Visualize detections for each class
        CONF_THRESH = 0.6
        NMS_THRESH = 0.3
        for ind, cls in enumerate(classes):
            cls_ind = CLASSES.index(cls)
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            keep = np.where(cls_scores >= CONF_THRESH)[0]
            cls_boxes = cls_boxes[keep, :]
            cls_scores = cls_scores[keep]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            print "keeped num:" , len(keep)
            dets = dets[keep, :]

            # select the bbox with highest score, calculate overlap
            if len(dets[:,-1]) ==0:
                print "miss!!"
                miss_num += 1
                missed_frame.append(imdb.image_path_at(i).split('/')[-1])
                missed_frame_ind.append(i)
                continue
            MaxI = np.argmax(dets[:,-1]) # find the maxium score
            score = dets[MaxI, -1]
            bbox = dets[MaxI, :4]
            bbox =[int(j) for j in bbox]
            gt = GT[i]['boxes'][ind]
            gt = [int(j) for j in gt]
            iou = IOU(bbox,gt)
            print "iou:",iou
            if iou < 0.5:
                print "low overlap"
                miss_num +=1
                missed_frame.append(imdb.image_path_at(i).split('/')[-1])
                missed_frame_ind.append(i)

            vis_detections(im, cls,bbox,score,gt, thresh=CONF_THRESH)
    print "error frame list:",missed_frame
    output_dir = os.path.join(get_output_dir(imdb,net),"fail_detected_frames.txt")
    with open(output_dir,'w') as f:
        for i in range(len(missed_frame)):
            f.write("{}\t{}\n".format(missed_frame_ind[i],missed_frame[i]))
    print "number of miss frame:",miss_num
    print "accuracy:",1-(float(miss_num)/num_images)
    

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg_cnn_m_1024]',
                        choices=NETS.keys(), default='vgg_cnn_m_1024')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
                            'test.prototxt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'output', cfg.EXP_DIR,'train',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/scripts/'
                       'fetch_fast_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(caffemodel))[0]
    print '\n\nLoaded network {:s}'.format(caffemodel)


    demo(net, ('vehicle',))

    plt.show()
