#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=32G
#SBATCH --job-name=CNNTrain
#SBATCH --output=log/make_train.out
#SBATCH --error=log/make_train.err

module load anaconda

model_dir=''
model_name=''

# python make_train.py -d data/train.csv -v data/valid.csv -i data/image/ -mc model/$model_dir/$model_name.json -mw model/$model_dir/$model_name -lr 0.00005 -s True
# python make_train.py -d data/train.csv -v data/valid.csv -i data/image/ -lr 0.00005 -s True
# python make_train.py -d data/train.csv -v data/valid.csv -i data/image/ -clr 5e-6 -s True -mc model/$model_dir/$model_name.json -mw model/$model_dir/$model_name

# python make_train.py -d data/train.csv -v data/valid.csv -i data/image/ -mc model/$model_dir/$model_name.json -mw model/$model_dir/$model_name -o SGD -s True
# python make_train.py -d data/train.csv -v data/valid.csv -i data/image/ -clr 1e-5 -s True -mc model/$model_dir/$model_name.json -mw model/$model_dir/$model_name -o SGD
python make_train.py -d data/train.csv -v data/valid.csv -i data/image/ -lr 0.01 -s True -o SGD