#!/bin/bash

set -x
set -e

time ./tools/demo.py --gpu 0 \
  --net vgg_cnn_m_1024
