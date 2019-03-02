# Neural network pruning

This repo provides an investigation of two methods of pruning neural networks:
weight pruning and neuron pruning. An MLP is trained on MNIST and pruned with
various fractions of weights/neurons left in the network. Results in terms of
prediction accuracy and inference time are plotted.

The code has been tested on Python 3.6 and TensorFlow 1.12. To install
the dependencies:

```
pip install -r requirements.txt
```

To run:

```
python main.py
```

Note that it can take several minutes to run, as inference is done on CPU to
get accurate measurement.
