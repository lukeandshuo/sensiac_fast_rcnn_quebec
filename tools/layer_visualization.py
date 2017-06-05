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
from skimage import img_as_ubyte
from skimage import exposure

Type =cfg.EXP_DIR
CLASSES = ('__background__',
           'vehicle')

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}


def vis_detections(i,im, class_name, bbox,score, gt,thresh=0.5):
    """Draw detected bounding boxes."""
   # inds = np.where(dets[:, -1] >= thresh)[0]
    cv2.putText(im,'{:s} {:.3f}'.format(class_name,score),(bbox[0],bbox[1]-3),0,0.6,(255,255,255))
    cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),2)
   # cv2.rectangle(im, (gt[0], gt[1]), (gt[2], gt[3]), (255,, 0), 2)
    cv2.imshow("result",im)
    cv2.imwrite(str(i) + "_" + Type + "_" + "det.png",im)
    cv2.waitKey(0)

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


def vis_square(data,i, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    # print data.shape
    # cv2.imshow("filter",data)
    # cv2.waitKey(0)
    plt.figure()
    plt.imshow(data)
    plt.imsave(str(i)+"_"+Type+"_"+"conv5.png",data)
    plt.show()
def vis_square_single(data,i, padsize=0, padval=0):

    # force the number of filters to be square
    data = data[np.newaxis,:,:]
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    # print data.shape
    # cv2.imshow("filter",data)
    # cv2.waitKey(0)
    plt.figure()
    plt.imshow(data)
    plt.imsave(str(i)+"_"+Type+"_"+"conv5.png",data)
    plt.show()
def _strech_intensity(img):
    # stretching  intensity
    img = exposure.rescale_intensity(img, in_range=(np.min(img), np.max(img)))
    # transform to char level
    img = img_as_ubyte(img)
    return img
def generateRGB(data,i=0, padsize=0, padval=0):
    print data.shape
    G = data[0,:]
    G = _strech_intensity(G)
    B = data[1,:]
    B = _strech_intensity(B)
    R = data[2,:]
    R = _strech_intensity(R)
    GBR = cv2.merge((G,B,R))
    cv2.imshow("featuremap",GBR)
    cv2.waitKey(0)

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
    print Type
    imdb = get_imdb('sensiac_test')
    num_images = len(imdb.image_index)
    roidb = imdb.roidb
    GT=imdb.gt_roidb()
    # i = 200,# the second image 200,240,(400,CONV5)
    i = 112 #401 fixed
    im = cv2.imread(imdb.image_path_at(i))
    print "raw image",im.shape
    cv2.imshow("raw",im)
    plt.figure()
    plt.imshow(im)
    plt.imsave(str(i)+"_"+Type+"_"+"raw.png",im)
    scores, boxes = im_detect(net, im, roidb[i]['boxes'])
    #
    for k,v in net.blobs.items():
        print (k,v.data.shape)
    ##visualized filter
    # filters= net.params['conv1'][0].data
    # print filters.shape
    # vis_square(filters.transpose(0,2,3,1))
    ## visualized blob
    # feat = net.blobs['conv5'].data[0,79]
    feat = net.blobs['fuse2'].data[0,:]
    print feat.shape
    # vis_square_single(feat,i, padval=1)
    # vis_square(feat,i, padval=1)
    # generateRGB(feat)
    max_per_image =2000

    CONF_THRESH = 0.0
    NMS_THRESH = 0.3
    for ind, cls in enumerate(classes):
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        print "lenlll:",cls_boxes.shape
        top_inds = np.argsort(-cls_scores)[:max_per_image]
        cls_scores = cls_scores[top_inds]
        cls_boxes = cls_boxes[top_inds, :]

        keep = np.where(cls_scores >= CONF_THRESH)[0]
        cls_boxes = cls_boxes[keep, :]
        cls_scores = cls_scores[keep]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        print "raw num:",len(dets)
        keep = nms(dets, NMS_THRESH)
        print "keeped num:" , len(keep)
        dets = dets[keep, :]
        MaxI = np.argmax(dets[:,-1]) # find the maxium score
        print "max indux",MaxI
        score = dets[MaxI, -1]
        bbox = dets[MaxI, :4]
        bbox =[int(j) for j in bbox]
        gt = GT[i]['boxes'][ind].astype(float)
        #gt = [int(j) for j in gt]
        iou = IOU(bbox,gt)
        print "iou:",iou
        vis_detections(i,im, cls,bbox,score,gt, thresh=CONF_THRESH)

if __name__ == '__main__':

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', 'Fusion_Net',
                            'test.prototxt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'output', Type,'train',
                 "Fusion_Net_fast_rcnn_iter_40000.caffemodel")

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/scripts/'
                       'fetch_fast_rcnn_models.sh?').format(caffemodel))

    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(caffemodel))[0]
    print '\n\nLoaded network {:s}'.format(caffemodel)


    demo(net, ('vehicle',))