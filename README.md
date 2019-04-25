# BCCNet
(C) Copyright 2019, University of Oxford. Written by [Olga Isupova](https://olgaisupova.github.io/)

This is an implementation of a Bayesian classifier combination neural network - a neural network that is trained directly from noisy crowdsourced labels, it jointly aggregates crowdsourced labels removing biases and identifying reliability of each crowd member and learns weights of a neural network, which then can be used to label new data.

The model is introduced in [O.Isupova, Y.Li, D.Kuzin, S.J.Roberts, K.Willis, and S.Reece, "BCCNet: Bayesian classifier combination neural network"](https://arxiv.org/abs/1811.12258), in NeurIPS 2018 Workshop on Machine Learning for the Developing World.

## Inference
Inference in the model is done by variational EM-algorithm:
* at the E-step the estimates of approximating posterior for the true labels and confusion matrices for each crowd member are updated -- the function VB_iteration in VariationalInference/VB_iteration.py
* at the M-step parameters of a neural network are updated -- one epoch of a standard backpropagation with the cross-entropy loss and current approximating posterior for the true labels as a target

Note that any neural network for classification can be used

## Getting started
demo.py and demo.ipynb provide a demo of a BCCNet on MNIST:
* first generate synthetic crowdsourced labels for MNIST. Only 50% of training data is labelled by 4 crowd members with the average reliability of 0.6
* iterate one call of the VB_iteration function and one epoch of backpropagated updates for the parameters of a neural network
* save weights of the trained neural network
* plot accuracy performance 
