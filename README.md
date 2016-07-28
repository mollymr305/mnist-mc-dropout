# mnist-mc-dropout


# Description

Writing Python (Lasagne + Theano library) code for respresenting model uncertainty in deep learning. Based on the following:

* 2016: [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142), [Dropout as a Bayesian Approximation: Appendix](https://arxiv.org/abs/1506.02157)

Training takes about less than 3 hours (for 3000 epochs) minutes with GPU; thanks [National Supercomputing Centre (NSCC) Singapore](http://www.nscc.sg)!

The main implementation is in ```mnist_mc_dropout.py``` which uses helper functions from ```__helpers__.py```, and of course the dataset ```mnist.pkl.gz```. For plotting training/validation errors see ```plot_error.py```. All the outputs are saved/pickled in the ```output``` folder.

Run/theano settings: ```THEANO_FLAGS='mode=FAST_RUN, device=gpu, floatX=float32' python mnist_mc_dropout.py```


# (Some) Results:

## Training details...

<img src="./output/errors.jpg">

## Interesting cases...

<img src="./output/index_8.jpg" height="200" width="400"> <img src="./output/index_15.jpg" height="200" width="400">

<img src="./output/index_18.jpg" height="200" width="400"> <img src="./output/index_20.jpg" height="200" width="400">

<img src="./output/index_35.jpg" height="200" width="400"> <img src="./output/index_36.jpg" height="200" width="400">

<img src="./output/index_62.jpg" height="200" width="400"> <img src="./output/index_65.jpg" height="200" width="400">

<img src="./output/index_73.jpg" height="200" width="400"> <img src="./output/index_78.jpg" height="200" width="400">

<img src="./output/index_92.jpg" height="200" width="400"> <img src="./output/index_95.jpg" height="200" width="400">