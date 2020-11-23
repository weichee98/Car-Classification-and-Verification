#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=32G
#SBATCH --job-name=CNNTest
#SBATCH --output=log/make_test.out
#SBATCH --error=log/make_test.err

module load anaconda

model_dir='inception_v3_make_classifier2'
model_name='inception_v3_make_classifier2_20201017215045'

python make_test.py -d data/test.csv -i data/image/ -mc model/$model_dir/$model_name.json -mw model/$model_dir/$model_name