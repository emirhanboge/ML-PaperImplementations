# ImageNet Classification with Deep Convolutional Neural Networks

This is a [PyTorch](http://pytorch.org/) implementation of the [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) architecture.

## Paper Review

This paper is published in NIPS 2012. It is the first work that popularized Convolutional Neural Networks in Computer Vision. The authors of this paper are Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton. The main contribution of this paper is the introduction of the AlexNet architecture, which is a deep convolutional neural network that has 8 layers, 5 convolutional layers, and 3 fully connected layers. The architecture is trained on the ImageNet dataset and achieves state-of-the-art performance on the ImageNet classification task. The paper also introduces the practical use of ReLU activation function and the dropout regularization technique. 

### Architecture

You can find the architecture code in `alexnet.py`. The architecture is as follows:

1. Convolutional Layer 1: 96 filters of size 11x11 with stride 4 and ReLU activation
2. Max Pooling Layer 1: 3x3 filters with stride 2
3. Convolutional Layer 2: 256 filters of size 5x5 with padding 2 and ReLU activation
4. Max Pooling Layer 2: 3x3 filters with stride 2
5. Convolutional Layer 3: 384 filters of size 3x3 with padding 1 and ReLU activation
6. Convolutional Layer 4: 384 filters of size 3x3 with padding 1 and ReLU activation
7. Convolutional Layer 5: 256 filters of size 3x3 with padding 1 and ReLU activation
8. Max Pooling Layer 3: 3x3 filters with stride 2
9. Fully Connected Layer 1: 4096 units with ReLU activation
10. Dropout Layer 1: Dropout with probability 0.5
11. Fully Connected Layer 2: 4096 units with ReLU activation
12. Dropout Layer 2: Dropout with probability 0.5
13. Fully Connected Layer 3: 1000 units (number of classes in ImageNet) with Softmax activation

### Training

The model is trained on the ImageNet dataset. The input images are resized to 256x256 and then a 224x224 crop is randomly sampled from the resized image. The model is trained using stochastic gradient descent with a batch size of 128, momentum of 0.9, and weight decay of 0.0005. The learning rate is initialized to 0.01 and is reduced by a factor of 10 when the validation error plateaus. The model is trained for 90 epochs. Because of the large size of the ImageNet dataset, CIFAR-10 is used for training and testing the model in this repository for demonstration purposes.

## Usage 

You can train the model using the following command:

```bash
python train.py
```

## References

1. [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
2. [ImageNet](http://www.image-net.org/)
3. [PyTorch](http://pytorch.org/)
```
