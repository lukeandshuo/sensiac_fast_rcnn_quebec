#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/train_fusion_net.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu 0 \
 --solver models/Fusion_Net/solver.prototxt \
 --imdb sensiac_train \
 --iters 40000


