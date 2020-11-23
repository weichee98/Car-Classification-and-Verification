#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=32G
#SBATCH --job-name=Triplet
#SBATCH --output=log/make_triplet.out
#SBATCH --error=log/make_triplet.err

module load anaconda

model_dir='triplet_inception_v3_hard'
model_name='triplet_inception_v3_hard_20201114220242'

python make_triplet.py -d data/train.csv -v data/valid.csv -i data/image/ -mc model/$model_dir/$model_name.json -mw model/$model_dir/$model_name -lr 5e-6 -o ADAM -md HARD
# python make_triplet.py -d data/train.csv -v data/valid.csv -i data/image/ -lr 2e-5 -o ADAM -md HARD