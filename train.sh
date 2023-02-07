#!/usr/bin/env bash
MVS_TRAINING="/mnt/bn/yjb01/mvs/train/mvs_training/dtu/"
python3 train.py --dataset=dtu_yao --batch_size=4 --trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt --numdepth=192 --logdir /mnt/bn/yjb01/mvs/checkpoints/d192 --resume $@
