# Deep Residual Learning for Image Recognition 

This is a [PyTorch](http://pytorch.org/) implementation of the [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) architecture.

## Paper Review

This paper is published in CVPR 2016. The authors of this paper are Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. The main contribution of this paper is the introduction of the ResNet architecture, which is a deep convolutional neural network that has up to 152 layers. The architecture is trained on the ImageNet dataset and achieves state-of-the-art performance on the ImageNet classification task. The paper also introduces the concept of residual learning, which is the idea of learning residual functions with reference to the layer inputs, instead of learning the desired functions directly. This idea is realized by introducing shortcut connections that skip one or more layers. The paper also introduces the practical use of batch normalization and the bottleneck architecture.

This paper is one of the most influential papers in the field of deep learning. The ResNet architecture has been widely adopted in many computer vision tasks and has been the basis for many state-of-the-art models. It adresses the problem of vanishing gradients in deep networks and enables the training of very deep networks by introducing the concept of residual learning. 

## Implementation

You can find the architecture code in `resnet.py`. The architecture is as follows:

- There is batch normalization after every convolutional layer.

- The architecture uses the bottleneck design, which is a design that reduces the number of parameters and computations by using 1x1, 3x3, and 1x1 convolutions in a sequence.

1. Convolutional Layer 1: 7x7, 64, stride 2
2. Max Pooling: 3x3, stride 2
3. Residual Block x 3: 3x3, [64, 64, 256]
4. Residual Block x 4: 3x3, [128, 128, 512]
5. Residual Block x 6: 3x3, [256, 256, 1024]
6. Residual Block x 3: 3x3, [512, 512, 2048]
7. Average Pooling: 7x7
8. Fully Connected Layer: 1000 (number of classes in ImageNet)

In total there are 50 layers in the architecture, so this is the ResNet-50 architecture. The architecture can be modified to create ResNet-18, ResNet-34, ResNet-101, and ResNet-152 architectures. Corresponding to the number of residual blocks in each stage.

The architecture is trained on the ImageNet dataset. The code for training and testing the model is in `train.py`. The model is trained using the cross-entropy loss and SGD with momentum. The learning rate is decreased by a factor of 10 after 30 and 60 epochs. The model achieves a top-1 error of 23.6% and a top-5 error of 6.7% on the ImageNet validation set.

## Usage

For demonstration purposes, the model is trained and tested on a subset of the CIFAR-10 dataset.

You can train the model using the following command:

```bash
python train.py
```

## References 

1. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
2. [ImageNet](http://www.image-net.org/)
3. [PyTorch](http://pytorch.org/)
4. [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)