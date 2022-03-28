# equivariant-convolutions

This repo contains the code to replicate the experiments of our CVPR2022 paper [Enabling Equivariance for Arbitrary Lie Groups](https://arxiv.org/abs/2111.08251).  In particular:
 - lie.py contains the code to establish a LieGroup class, which corresponds to either the affine or homography groups.  A LieGroup class instantiates the Lie algebra basis described in the Preliminaries section of the [paper](https://arxiv.org/abs/2111.08251), with corresponding basis of the adjoint representation, as well as functions to compute the exponential and action of a Lie algebra vector on two-dimensional Euclidean space.
 - sampling.py contains the code used to generate samples in parallel from right Haar measure given a LieGroup, as described in Section 4.2 of the [paper](https://arxiv.org/abs/2111.08251).
 - layers.py contains the code used for the Lie group convolutional layers described in Sections 4.1 and 4.2 of the [paper](https://arxiv.org/abs/2111.08251).
 - data.py contains the code we used to create PyTorch datasets from the [homNIST](https://www.kaggle.com/datasets/lachlanemacdonald/homnist) and [affNIST](https://www.cs.toronto.edu/~tijmen/affNIST/) .mat files.
 - experiment.py is the code to train a G-equivariant convolutional model on (padded) MNIST, to then test on GNIST (where G denotes either the affine or homography group).
 - experiment_e2.py is the code we used to benchmark the E2SFCNN E(2)-equivariant convolutional model of [General $E(2)$-Equivariant Steerable CNNs](https://arxiv.org/abs/1911.08251) on affNIST and homNIST after training on (padded) MNIST.  Running this file requires the [e2sfcnn.py file](https://github.com/QUVA-Lab/e2cnn_experiments/blob/master/experiments/models/e2sfcnn.py), as well as the [e2cnn library](https://github.com/QUVA-Lab/e2cnn).  The model is trained with the [recommended hyperparameters](https://github.com/QUVA-Lab/e2cnn_experiments/blob/master/experiments/mnist_bench_single.sh).
 - create_homNIST.py is the code we used to generate the homNIST test set.

To replicate our experiments using the experiment.py file:
 - From [this link](https://www.cs.toronto.edu/~tijmen/affNIST/32x/just_centered/), download and extract into local folder /data the files: training.mat.zip as training.mat, and test.mat.zip as test.mat.  The files training.mat and test.mat contain the black-padded MNIST training and test sets respectively.
 - From [this link](https://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/), download and extract into local folder /data the file test.mat.zip as affNIST_test.mat.  The file affNIST_test.mat contains the affNIST test set.
 - From [this link](https://www.kaggle.com/datasets/lachlanemacdonald/homnist), download the homNIST.mat file.  It contains the homNIST test set.

The experiment consists of training a model on (padded) MNIST, and then testing on either affNIST or homNIST.  To run the former, in experiment.py simply set group = 'affine'; for the latter set group = 'homography'.
