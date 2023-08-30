---
layout: page
title: "Neural Network (Self-)Tutorial"
background: "/img/MnistExamplesModified.png"
---

Back in 2017, I took Andrew Ng's Coursera class in machine learning.  I was inspired to better establish my knowledge of what he covered.

The [code](https://github.com/snyderjo/Neural-Network-Tutorial) contained represents my success at creating and estimating fully-connected neural network models (however inefficient).  This firmed up my knowledge of:
* python
* numpy
* neural networks--the fully connected (a.k.a. dense) variety
* backpropigation

I created the following modules:
* `inputLayers`
* `hiddenLayers`
* `outputLayers`
* `activation_functions`

These are called in the module
* `neuralNets`

All of the above are packaged into `nnfiles`, which is called in `MNIST test.ipynb` to estimate a fully-connected neural network on the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database).  I was able to achieve a 95% accuracy!

This was among my first projects in python and git.

Several years later, I added the code necessary to make each file module and provided a virtual environment (along with a handful of necessary updates).