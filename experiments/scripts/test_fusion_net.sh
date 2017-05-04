#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/test_fusion_net.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_net.py --gpu 0 \
  --def models/Fusion_Net/test.prototxt \
  --imdb sensiac_test
