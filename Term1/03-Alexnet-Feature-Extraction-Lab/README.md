# AlexNet Feature Extraction
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This lab guides you through using AlexNet and TensorFlow to build a feature extraction network.

## Setup
Before you start the lab, you should first install:
* Python 3
* TensorFlow
* NumPy
* SciPy
* matplotlib

# Class Instructions

## Transfer Learning with TensorFlow

**Transfer learning** is the practice of starting with a network that 
has already been trained, and then applying that network to your own problem.

Because neural networks can often take days or even weeks to train, 
transfer learning (i.e. starting with a network that somebody else 
has already spent a lot of time training) can greatly shorten training time.

So how do we apply transfer learning? Two popular methods are **feature extraction** and **finetuning**.

1. **Feature extraction**. We take a pretrained neural network and replace the final layer 
(classification layer) with a new classification layer for the new dataset or perhaps even 
a small neural network (eventually has a classification layer). During training the weights 
in all the old layers are frozen, only the weights for the new layer we added are trained. 
In other words, the gradient doesn't flow backwards past the first new layer we add.

2. **Finetuning**. Essentially the same as feature extraction except the weights of the old 
model aren't frozen. The network is trained end-to-end.  

In this lab and the one later in the section we're going to focus on feature 
extraction since it's less computationally intensive.

## Getting Started

1. Download the training data and AlexNet weights. These links are also available at the bottom of this page under Supporting Materials.

2. Clone the repository containing the code.

	* git clone https://github.com/udacity/CarND-Alexnet-Feature-Extraction 
	* cd CarND-Alexnet-Feature-Extraction

3. Make sure the downloaded files from step (1) are in the code directory as the code.

4. Open the code in your favourite editor.

## Feature Extraction via AlexNet

Here, you're going to practice feature extraction with AlexNet.

AlexNet is a popular base network for transfer learning because its structure is relatively straightforward, 
it's not too big, and it performs well empirically.

There is a TensorFlow implementation of AlexNet (adapted from Michael Guerhoy and Davi Frossard) in alexnet.py. 
You're not going to edit anything in this file but it's a good idea to skim through it to see how AlexNet is defined in TensorFlow.

## Credits

This lab utilizes:

	* An implementation of AlexNet created by [Michael Guerzhoy and Davi Frossard](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/)
	* AlexNet weights provided by the [Berkeley Vision and Learning Center](http://bvlc.eecs.berkeley.edu/)
	* Training data from the [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news)

AlexNet was originally trained on the [ImageNet database](http://www.image-net.org/).
