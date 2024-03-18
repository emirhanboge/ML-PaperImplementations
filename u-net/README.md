# U-Net: Convolutional Networks for Biomedical Image Segmentation

This is a [PyTorch](http://pytorch.org/) implementation of the [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) architecture.

## Paper Review

This paper was first introduced in the Springer MICCAI in 2015. The authors of this paper are Olaf Ronneberger, Philipp Fischer, and Thomas Brox. The main contribution of this paper is the introduction of the U-Net architecture, which is a deep convolutional neural network that has an encoder-decoder structure. The architecture is designed for biomedical image segmentation tasks. The architecture is trained on the ISBI cell tracking challenge dataset and achieves state-of-the-art performance on the dataset. 

### Implementation

First layers increase the number of channels and decrease the spatial dimensions, which is the encoder part. The second part of the network is the decoder, which increases the spatial dimensions and decreases the number of channels. The skip connections are used to concatenate the feature maps from the encoder to the decoder. The skip connections help the network to recover the spatial information that is lost during the encoding process. The network is trained using the binary cross-entropy loss function. 

Data augmentation is used to increase the size of the training dataset. The authors use elastic deformations, rotations, and horizontal flips to augment the dataset. 

Encoder, contracting path, consists of two 3x3 convolutions that are followed by ReLU units and a 2x2 max pooling with stride 2. The decoder, expansive path, consists of an upsampling of the feature map followed by a 2x2 convolution. The feature map is then concatenated with the corresponding feature map from the contracting path. The concatenated feature map is then followed by two 3x3 convolutions and ReLU units. In the last layer, a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes. In total there are 23 convolutional layers in the network.

Softmax is used to calculate the probability of each pixel belonging to a class. Since we try to segment the image into two classes, the probability of each pixel belonging to the class is calculated.

U-Net implementation is available in the `u_net.py` file. In this implementation VOCSegmentation dataset is used to train the network. The dataset is available in the torchvision package. The dataset consists of images and masks. The images are the input to the network and the masks are the ground truth. The masks are used to calculate the loss and to evaluate the performance of the network.

## Usage

To train the network, run the following command:

```bash
python train.py
```

## References

1. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
2. [PyTorch](http://pytorch.org/)
3. [VOCSegmentation](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html)
