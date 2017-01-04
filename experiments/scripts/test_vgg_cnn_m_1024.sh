#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/test_vgg_cnn_m_1024.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_net.py --gpu 0 \
  --net output/default/train/vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel \
  --imdb sensiac_test
