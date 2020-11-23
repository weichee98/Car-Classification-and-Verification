# Car Maker Classification

## Googlenet

Googlenet is a type of CNN network with 22 layers introduced in 2014. It allows the network to choose between multiple convolutional filter sizes in each block and have max-pooling layers in between. However, the model is losing trend because of improvement achieved in other CNN-based models.

![googlenet architecture](https://paperswithcode.com/media/methods/Screen_Shot_2020-06-22_at_3.28.59_PM.png)

## Inception V3

Inception V3 is an Inception model introduced in 2015 and developed based on previous Inception models. The main improvements include Label Smoothing, factorized 7 x 7 convolutions and auxiliary classifier etc.

![inception v3 architecture](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/05/inceptionv3.png)

## Improvements on Previous Models

### Googlenet to Classify More Car Maker Classes

The [previous work](https://arxiv.org/pdf/1506.08959.pdf) only exploits part of the datasets to classify cars into 75 makers. Here, we take the whole dataset to perform classification into 163 car makers. We use GoogLeNet with an output softmax layer of 163 classes.

|![Graph showing accuracy against epoch of GoogLeNet](../image/googlenet_accuracy.png)|![Graph showing loss against epoch of GoogLeNet](../image/googlenet_loss.png)|
|-|-|

The final accuracy obtained is 0.628 which is lower than the one published (0.829). This might be due to a greater number of classification classes.

### Inception V3 replacing Googlenet

Inception V3 has better performance in the computer vision field compared to GoogLeNet, here we replace Googlenet with Inception V3 and compare their performances on car maker classification.

| Model | Layers |
|-|-|
| Inception V3 (1) | <li>Layer 1: Inception V3 <li>Layer 2: Output Softmax Layer |
| Inception V3 (2) | <li>Layer 1: Inception V3 <li>Layer 2: Dropout (0.2 drop rate) <li>Layer 3: Output Softmax Layer |
| Inception V3 (3) | <li>Layer 1: Inception V3 <li>Layer 2: Dropout (0.2 drop rate) <li>Layer 3: 1000 Neurons Hidden Layer <li>Layer 4: Output Softmax Layer |
| Inception V3 (4) | <li>Layer 1: Inception V3 <li>Layer 2: Dropout (0.2 drop rate) <li>Layer 3: 1000 Neurons Hidden Layer <li>Layer 4: Dropout (0.2 drop rate) <li>Layer 5: 1000 Neurons Hidden Layer <li>Layer 6: Output Softmax Layer |

| Model | Accuracy | Loss |
|-|-|-|
| Inception V3 (1) | ![](../image/inception_v31_make_accuracy.png) | ![](../image/inception_v31_make_loss.png) |
| Inception V3 (2) | ![](../image/inception_v32_make_accuracy.png) | ![](../image/inception_v32_make_loss.png) |
| Inception V3 (3) | ![](../image/inception_v33_make_accuracy.png) | ![](../image/inception_v33_make_loss.png) |
| Inception V3 (4) | ![](../image/inception_v34_make_accuracy.png) | ![](../image/inception_v34_make_loss.png) |

We can see that after replacing Googlenet with Inception V3, the classification model managed to improve drastically.

### Comparison

| ![](../image/maker_classifier_accuracy.png) | ![](../image/maker_classifier_loss.png) |
|-|-|

We can observe that Inception V3 performs significantly better than GoogLeNet  in terms of test accuracies and loss. Inception V3 is deeper (42 layers) and it has more parameters than GoogLeNet, more features can be learnt from the network, resulting in higher accuracy. 

| ![](../image/make_classifier_accuracy_epoch.png) | ![](../image/make_classifier_loss_epoch.png) |
|-|-|

Comparing Inception V3 (1) with Inception V3 (2), the dropout layer does not help in improving test accuracy but it results in a lower test loss. Introducing a hidden layer of 1000 neurons in Inception V3 (3) improves the accuracy and lowers the loss as compared to Inception V3 (2). When 2 hidden layers of 1000 neurons are added in Inception V3 (4), the accuracies and losses become worse, which might be due to overfitting.

### Conclusion

As the model becomes deeper, the classification performance might be improved, but will also introduce more complexity to the model, results in overfitting. Therefore, regularizers and dropout layers can play significant role in helping the generalization of complex model, leading to better performance of deeper neural networks.

