#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=32G
#SBATCH --job-name=Dot
#SBATCH --output=log/make_dot.out
#SBATCH --error=log/make_dot.err

module load anaconda

model_dir='dot_inception_v3'
model_name='dot_inception_v3_20201119095045'

python make_dot.py -d data/train.csv -v data/valid.csv -i data/image/ -mc model/$model_dir/$model_name.json -mw model/$model_dir/$model_name -lr 1e-3 -o SGD
# python make_dot.py -d data/train.csv -v data/valid.csv -i data/image/ -lr 1e-3 -o SGD