#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=32G
#SBATCH --job-name=CNNTrain
#SBATCH --output=log/model_train.out
#SBATCH --error=log/model_train.err

module load anaconda

model_dir='inception_v3_model_classifier9'
model_name='inception_v3_model_classifier9_20201120133410'

# python model_train.py -d data/train.csv -v data/valid.csv -i data/image/ -lr 0.001 -s True -o SGD

# change learning rate
python model_train.py -d data/train.csv -v data/valid.csv -i data/image/ -clr 0.002 -s True -mc model/$model_dir/$model_name.json -mw model/$model_dir/$model_name -o SGD

# change learning rate and no lr scheduler
# python model_train.py -d data/train.csv -v data/valid.csv -i data/image/ -clr 0.001 -s True -mc model/$model_dir/$model_name.json -mw model/$model_dir/$model_name

# not changing learning rate
# python model_train.py -d data/train.csv -v data/valid.csv -i data/image/ -s True -mc model/$model_dir/$model_name.json -mw model/$model_dir/$model_name -o SGD

# not changing learning rate and no lr scheduler
# python model_train.py -d data/train.csv -v data/valid.csv -i data/image/ -mc model/$model_dir/$model_name.json -mw model/$model_dir/$model_name