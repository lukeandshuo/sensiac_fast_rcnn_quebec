# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import cPickle
import numpy as np
import matplotlib.pyplot as plt

def parse_rec(filename,ind):
    """ Parse a PASCAL sensiac xml file """
    objects = []

    with open(filename) as f:
            for i,line in enumerate(f):
                if i == ind:
                    line = line.strip().split(",")
                    x1 = float(line[0])-5
                    y1 = float(line[1])-5
                    x2 = float(line[2])+5
                    y2 = float(line[3])+5          
                    boxes = [x1,y1,x2,y2]
                    objects.append({'bbox': boxes, 'name': 'vehicle'})
                    break
    return objects

def sensiac_ap(rec, prec, use_07_metric=False):
    """ ap = sensiac_ap(rec, prec, [use_07_metric])
    Compute sensiac AP given precision and recall.
    If use_07_metric is true, uses the
    sensiac 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        plt.plot(mrec,mpre)
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.xlabel('recall')
        plt.ylabel('precision')
	plt.show()
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def sensiac_eval_ap(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = sensiac_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL sensiac evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use sensiac07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath,i)
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
	# print "R",R
        bbox = np.array([x['bbox'] for x in R])
	# print "bbox",bbox
        det = [False] * len(R)
        npos = npos + 1
        class_recs[imagename] = {'bbox': bbox,
                                 'det': det}
    # print "npos:",npos
    print "image length:",len(imagenames)
    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    #print "Score:",sorted_scores
    BB = BB[sorted_ind, :]
    #print "BBOX:",BB
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = sensiac_ap(rec, prec, use_07_metric)

    return rec, prec, ap

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

def vis_detections(im, class_name, bbox,score, gt,thresh=0.5):
    """Draw detected bounding boxes."""
   # inds = np.where(dets[:, -1] >= thresh)[0]
    cv2.putText(im,'{:s} {:.3f}'.format(class_name,score),(bbox[0],bbox[1]-3),0,0.6,(255,255,255))
    cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),2)
   # cv2.rectangle(im, (gt[0], gt[1]), (gt[2], gt[3]), (255,, 0), 2)
    cv2.imshow("result",im)
    cv2.waitKey(0)

def sensiac_eval_top1( output_dir,
            detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,):

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # if not os.path.isfile(cachefile):
        # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_rec(annopath,i)
        if i % 100 == 0:
            print 'Reading annotation for {:d}/{:d}'.format(
                i + 1, len(imagenames))
    # save
    print 'Saving cached annotations to {:s}'.format(cachefile)
    with open(cachefile, 'w') as f:
        cPickle.dump(recs, f)
    # else:
    #     # load
    #     with open(cachefile, 'r') as f:
    #         recs = cPickle.load(f)

    # extract gt objects for this class

    class_recs = {}
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        # print "bbox",bbox 
        class_recs[imagename] = {'bbox': bbox}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = np.array([x[0] for x in splitlines])
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    # print "BB",BB.shape
    # print confidence.shape
    missed_frame = []
    missed_frame_ind = []
    miss_num =0
    for image_ind, image_name in enumerate(imagenames):
        ids = np.where(image_ids == image_name)[0]
        boxes = BB[ids,:]
        scores = confidence[ids]
        gt = class_recs[image_name]['bbox'][0].astype(float)
        # print len(ids)
        if len(ids) == 0:
            print "not found missed"
            miss_num += 1
            missed_frame += [image_name]
            continue
        # print boxes,scores,gt

        max_ind = np.argmax(scores)
        score = scores[max_ind]
        bbox = boxes[max_ind]
        iou = IOU(bbox,gt)
        if image_ind == 821:
            print max_ind
            print iou
        # print iou
        if iou < 0.5:
            
            miss_num += 1
            missed_frame += [image_name]
            print "low overlap missed",image_ind,image_name
    print miss_num
    missed_frame_file = os.path.join(output_dir,"missed_frame.txt")
    with open(missed_frame_file,'w') as f:
        for i in missed_frame:
            f.write("{}\n".format(i))


    # print missed_frame
    
    top1_accuracy = 1-(float(miss_num)/len(imagenames))


    return top1_accuracy