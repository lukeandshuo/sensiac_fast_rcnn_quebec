# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.sensiac
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from sensiac_eval import sensiac_eval_ap, sensiac_eval_top1
from fast_rcnn.config import cfg
class sensiac(datasets.imdb):
    def __init__(self, image_set, devkit_path):
        datasets.imdb.__init__(self, image_set)
        self._image_set = image_set #train or test
        self._devkit_path = devkit_path
        self._data_path = os.path.join(self._devkit_path)
        self._classes = ('__background__', # always index 0
                         'vehicle')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = ['.png']
	self._image_type = cfg.EXP_DIR
        self._image_index = self._load_image_set_index()
  	self._comp_id = 'comp1'
        self._salt = str(uuid.uuid4())
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
	
        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), \
                'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        for ext in self._image_ext:
            image_path = os.path.join(self._data_path, 'Imagery',self._image_type,'images',index + ext)
            if os.path.exists(image_path):
                break
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
	return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt
        image_set_file = os.path.join(self._data_path, 'Train_Test', self._image_type,
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self._load_sensiac_annotation()
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        print "ss roi"

        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set != 'test_real':
            print 'yes'
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
	with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        print "load ss roi"
        filename = os.path.abspath(os.path.join(self._devkit_path,'ROI',self._image_type,
                                                 self._image_set+ '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
	raw_data = sio.loadmat(filename)['all_boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            # box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1) # change to x1,y1,x2,y2 !!! notice for edgebox!
            box_list.append(raw_data[i][:,:]-1)

	return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_IJCV_roidb(self):
        """
        eturn the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
                format(self.name, self.config['top_k']))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_IJCV_roidb(self, gt_roidb): # dont use
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 self.name))
        assert os.path.exists(IJCV_path), \
               'Selective search IJCV data not found at: {}'.format(IJCV_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:top_k, :]-1).astype(np.uint16))

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_sensiac_annotation(self):
        """
        Load image and bounding boxes info from txt files of vehicle.
        """
        filename = os.path.join(self._data_path, 'Annotations',self._image_type, self._image_set + '.txt')
        print filename
        # print 'Loading: {}'.format(filename)
        gt_roidb = []
        with open(filename) as f:
            for line in f:
                line = line.strip().split(",")
                num_objs = 1
                boxes = np.zeros((num_objs, 4), dtype=np.uint16)

                gt_classes = np.zeros((num_objs), dtype=np.int32)
                overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
		seg_areas = np.zeros((num_objs),dtype=np.float32)
                for i in range(num_objs):
                    x1 = float(line[0 + i * 4])-5
                    y1 = float(line[1 + i * 4])-5
                    x2 = float(line[2 + i * 4])+5
                    y2 = float(line[3 + i * 4])+5
                    cls = self._class_to_ind['vehicle']
                    boxes[i,:] = [x1,y1,x2,y2]
                    gt_classes[i]=cls
                    overlaps[i,cls]=1.0
		    seg_areas[i]=(x2-x1+1)*(y2-y1+1)
                overlaps = scipy.sparse.csr_matrix(overlaps)
                gt_roidb.append({'boxes': boxes, 'gt_classes': gt_classes, 'gt_overlaps': overlaps, 'flipped': False,'seg_areas': seg_areas})
        return gt_roidb

    def _get_comp_id(self):
        return self._comp_id

    def _write_sensiac_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_sensiac_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return self._comp_id
    def _get_sensiac_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'

        path = os.path.join(
            self._devkit_path,
            'results',
            self._image_type,
            'Main')
	if not os.path.exists(path):
	    os.makedirs(path)
	template_path = os.path.join(path,filename)
        return template_path    

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'setenv(\'LC_ALL\',\'C\'); voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def _do_python_eval(self, output_dir = 'output'):
        annopath = filename = os.path.join(self._data_path, 'Annotations',self._image_type, self._image_set + '.txt')
        imagesetfile = os.path.join(self._data_path, 'Train_Test', self._image_type, self._image_set + '.txt')
        cachedir = os.path.join(self.cache_path, 'annotations_cache')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        # evaluated by AP
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = False 
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')

        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_sensiac_results_file_template().format(cls)
            rec, prec, ap = sensiac_eval_ap(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_ap.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))

        #evaluated by Top1
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_sensiac_results_file_template().format(cls)
            top1_a = sensiac_eval_top1(output_dir,filename,annopath,imagesetfile,cls,cachedir,ovthresh=0.5)
            print "top1 accuracy", top1_a
    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_sensiac_results_file(all_boxes)
        self._do_python_eval(output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__),'..','..','data','sample_data')
    d = datasets.sensiac('train', data_dir)
    res = d.roidb
    from IPython import embed; embed()
