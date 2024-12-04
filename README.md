Scripts to train a neural-network model for digit recognition using only numpy, written and commented in a way that is easy to understand.

The training dataset is [the MNIST dataset](http://yann.lecun.com/exdb/mnist/) of 60,000 handwritten digits, plus 10,000 for testing, and is included in the `data` folder.

The script improves on the [excellent tutorial by Samson Zhang](https://www.youtube.com/watch?v=w8yWXqWQYmU) in the following way:

- Arbitrary number of hidden layers
- He initialization for faster convergence
- Vector biases, rather than scalar
- Fix in softmax denominator

# Available scripts

- `scripts/main.py` runs the neural network with an arbitrary number of hidden layers
- `scripts/one-hidden-layer.py` runs the neural network with just one hidden layer.  Probably the best starting point to understand back propagation
- `scripts/samson.py` same script as in Samson's tutorial
- `scripts/show-mnist-image.py` shows specific images from the MNIST dataset (requires `datasets` package)

All of the training scripts will print the accuracy on the test set after training.

# Usage example

Run the following command to obtain an accuracy of around 92% on the test set:

```bash
python -m scripts.main --n_hidden 32 16 --epochs 500 --learning_rate 0.2 --initial_params he
```

Arguments:

- `n_hidden` is the number of neurons in each hidden layer, e.g. `--n_hidden 32 16` means two hidden layers with 32 and 16 neurons respectively
- `epochs` is the number of times the training set is passed through the network
- `learning_rate` is how much the weights are updated at each iteration
- `initial_params` is the initialization method for the weights.  Options are `he`, `gaussian`, `samson` (as in Samson's tutorial)


# Accuracy comparisons

Have fun and compare the accuracy of our script with the results shown in [Yan LeCun's website](https://yann.lecun.com/exdb/mnist/index.html) and in his [1998 paper](https://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf).

Although our model is simpler than most of those considered by LeCun (does not alter the MNIST training by distorting or deslanting images; does not use regularization of loss function; performs gradient average rather than stochastic descent), it achieves a comparable accuracy on the test set.

For example, running the script with a single 300-unit hidden layer yields an error of about 4.5% after 1000 passes, similarly to what LeCun obtains in section 3.C.5 of the paper:

```bash
python -m scripts.main --n_hidden 300 --learning_rate 0.2 --epochs 1000 --initial_params he
```

In section 3.C.6 LeCun achieves a 3.05% error by running a NN with two hidden layers of 300 and 100 units, respectively.  Our script yields a similar result with an error of 2.9%, with the following command:

```bash
python -m scripts.main --n_hidden 300 100 --learning_rate 0.2 --epochs 1000 --initial_params he
```

It seems there's still room for improvement, as doubling the number of passes to 2000 yields an error of 2.3% on the test set which further decreases to 1.8% for 5000 passes.


# Comparison with Samson's script

To run the exact script from [Samson tutorial](https://www.youtube.com/watch?v=w8yWXqWQYmU), run:

```bash
python -m scripts.samson
```

You will obtain an accuracy of around 85% on the test set.  The same result can be reproduced with either of the following commands:

```bash
python -m scripts.main --n_hidden 10 --learning_rate 0.1 --epochs 500 --initial_params samson
python -m scripts.one-hidden-layer --n_hidden 10 --learning_rate 0.1 --epochs 500 --initial_params samson
```
