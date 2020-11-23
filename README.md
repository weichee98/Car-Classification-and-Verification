# Car Classification and Verification

Fine-grained car classification and recognition can be used for various purposes especially in modern transportation systems such as regulation, description and indexing. The overall goal of this project is to classify an image into one of the 163 car makers and 1716 car models.

## Dataset

The [CompCars datasets](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html) have two scenarios which are images from web-nature and surveillance-nature images. For this project, we will only use the web-nature data that captures the entire car which is 136726 images in total. The images can be classified into 163 car makes with 1716 car models.

The dataset includes 4 unique features which are car hierarchy, car attributes, viewpoints and car parts. Car hierarchy contains information of car makers and car models that are the results of classification to be achieved. The other feature used will be the viewpoints. We used the viewpoints to split the data to ensure an even distribution of the different viewpoints of the same car model in test, train, validation data.

![front](image/front.png)
![rear](image/rear.png)
![side](image/side.png)
![front_side](image/front_side.png)
![rear_side](image/rear_side.png)
![uncertain](image/uncertain.png)


## Getting Started

### Dataset Download
Please refer to the download guide: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/instruction.txt

### Clone Repository
```
git clone https://github.com/weichee98/Car-Classification-and-Verification.git
```

### Requirements
- [numpy](https://numpy.org/install/)
- [scipy](https://www.scipy.org/install.html)
- [pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)
- [TensorFlow 2.2.0](https://www.tensorflow.org/install) or later
- [matplotlib](https://matplotlib.org/)
- [tqdm](https://github.com/tqdm/tqdm)
- [sklearn](https://scikit-learn.org/stable/)

## Directory Structure

The main directory in this project is [Human-Activity-Recognition](https://github.com/weichee98/Human-Activity-Recognition), which has the directory structure as below:
```
Car-Classification-and-Verification
├── data
│   └── image
|       └── 1
|           └── 1101
|               └── 2011
|                   └── 3a62131af5fe8e.jpg
|                   └── 07b90decb92ba6.jpg
|                   └── ...
|           └── 1102
|           └── ...
|       └── 2
|       └── ...
|   └── label
│   └── image
|       └── 1
|           └── 1101
|               └── 2011
|                   └── 3a62131af5fe8e.txt
|                   └── 07b90decb92ba6.txt
|                   └── ...
|           └── 1102
|           └── ...
|       └── 2
|       └── ...
│   └── misc
|       └── attributes.txt
|       └── car_type.mat
|       └── make_model_name.mat
|   └── train_test_split
|       └── classification
|           └── test.txt
|           └── train.txt
|       └── part
|           └── test_part_1.txt
|           └── test_part_2.txt
|           └── ...
|           └── train_part_1.txt
|           └── train_part_2.txt
|           └── ...
|       └── verification
|           └── verification_pairs_easy.txt
|           └── verification_pairs_hard.txt
|           └── verification_pairs_medium.txt
|           └── verification_train.txt
│   └── data.csv
|   └── result.csv
│   └── test.csv
│   └── train.csv
|   └── valid.csv
├── model
│   └── ...
│   └── googlenet
|       └── base
|           └── googlenet.h5
|           └── googlenet.json
|           └── googlenet.npy
|           └── googlenet.py
|       └── googlenet_make_classifier_20201001232148.data-00000-of-00002
|       └── googlenet_make_classifier_20201001232148.data-00001-of-00002
|       └── googlenet_make_classifier_20201001232148.index
|       └── googlenet_make_classifier_20201001232148.json
│   └── inception_v3_make_classifier
|       └── inception_v3_make_classifier_20201007001143.data-00000-of-00002
|       └── inception_v3_make_classifier_20201007001143.data-00001-of-00002
|       └── inception_v3_make_classifier_20201007001143.index
|       └── inception_v3_make_classifier_20201007001143.json
│   └── ...
│   └── model_manager.py
│   └── dot_make_description.json
│   └── make_classifier_description.json
│   └── model_classifier_description.json
│   └── triplet_make_description.json
├── src
│   └── dot_product.py
│   └── input_pipeline.py
│   └── prepare_dataset.py
│   └── train_test_split.py
│   └── triplet.py
├── utils
│   └── dict_json.py
│   └── logger.py
├── make_dot.py
├── make_test.py
├── make_train.py
├── make_triplet.py
└── model_test.py
└── model_train.py
```
