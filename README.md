# equivariant-convolutions

This repo contains the code to replicate the experiments of our CVPR2022 paper [Enabling Equivariance for Arbitrary Lie Groups](https://arxiv.org/abs/2111.08251).  In particular:
 - lie.py contains the code to establish a LieGroup class, which corresponds to either the affine or homography groups.  A LieGroup class instantiates the Lie algebra basis described in the Preliminaries section of the [paper](https://arxiv.org/abs/2111.08251), with corresponding basis of the adjoint representation, as well as functions to compute the exponential and action of a Lie algebra vector on two-dimensional Euclidean space.
 - sampling.py contains the code used to generate samples in parallel from right Haar measure given a LieGroup, as described in Section 4.2 of the [paper](https://arxiv.org/abs/2111.08251).
 - layers.py contains the code used for the Lie group convolutional layers described in Sections 4.1 and 4.2 of the [paper](https://arxiv.org/abs/2111.08251).
 - data.py contains the code we used to create PyTorch datasets from the [homNIST](https://www.kaggle.com/datasets/lachlanemacdonald/homnist) and [affNIST](https://www.cs.toronto.edu/~tijmen/affNIST/) .mat files.
 - experiment.py is the code to train a $G$-equivariant convolutional model on (padded) MNIST, to then test on GNIST (where G denotes either the affine or homography group).
 - experiment_e2.py is the code we used to benchmark the E2SFCNN E(2)-equivariant convolutional model of [General $E(2)$-Equivariant Steerable CNNs](https://arxiv.org/abs/1911.08251) on affNIST and homNIST.  Running this file requires the [e2sfcnn.py file](https://github.com/QUVA-Lab/e2cnn_experiments/blob/master/experiments/models/e2sfcnn.py). 
 - create_homNIST.py is the code we used to generate the homNIST test set.
