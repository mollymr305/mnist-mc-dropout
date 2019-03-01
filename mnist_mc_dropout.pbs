#!/bin/bash
#PBS -q gpu
#PBS -l walltime=3:00:00

cd ~
cd mnist-mc-dropout

module load cuda/7.5

THEANO_FLAGS='mode=FAST_RUN,device=gpu,floatX=float32' python mnist_mc_dropout.py