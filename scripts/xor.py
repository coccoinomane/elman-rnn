"""
Elman's simple recurrent neural network (RNN) for the XOR problem.

Differences from Elmans original paper:
- Using ReLU activation function instead of sigmoid
- Using logistic loss instead of squared error to simplify the backpropagation
"""

import numpy as np
from numpy.typing import NDArray
from numpy import ndarray, dtype, floating
from typing import Literal
from numpy.typing import NDArray

# Hyperparameters
# Please note that input and output layers both have only 1
# unit (the current and next bit, respectively)
n_passes = 100
n_units_hidden = 2
learning_rate = 0.1

# Dataset parameters
random_seed = 42
n_samples = 1000

# Set random seed for reproducibility
np.random.seed(42)

# Types
Matrix_2X2 = ndarray[tuple[Literal[2], Literal[2]], dtype[floating]]  # [2x2] matrix
Matrix_2X1 = ndarray[tuple[Literal[2], Literal[1]], dtype[floating]]  # [2x1] column vector
Matrix_1X2 = ndarray[tuple[Literal[1], Literal[2]], dtype[floating]]  # [1x2] row vector
Matrix_1X1 = ndarray[tuple[Literal[1], Literal[1]], dtype[floating]]  # [1x1] scalar
Sequence = NDArray[np.int8]  # sequence of bits

# Create input dataset: 3,000 bits where each third bit is the XOR of
# the previous two bits. For example: 100010011110101...
training_pairs = np.random.randint(0, 2, [n_samples, 2])
training_xors = np.logical_xor(training_pairs[:, 0], training_pairs[:, 1]).astype(int)
training_sequence: Sequence = np.concatenate(
    [training_pairs, training_xors[:, None]], axis=1
).flatten()  # [3000x1]


# It is best to represent the NN with a class because we need to keep
# track of the timesteps and of the context units
class ElmanRNN:

    def __init__(self, n_hidden: int) -> None:
        # Assignments
        self.n_hidden = n_hidden

        # Types
        self.context: Matrix_2X1
        self.W1: Matrix_2X1
        self.WC: Matrix_2X2
        self.W2: Matrix_1X2

        # Initializations
        self.reset_context()
        self.init_params()

    def reset_context(self) -> None:
        """Set the context units to 0.5 as per the Elman paper"""
        self.context = np.full((self.n_hidden, 1), 0.5)  # [2x1]

    def init_params(self) -> None:
        """Initialize the model parameters at t0"""
        self.W1 = np.random.rand(self.n_hidden, 1)  # input -> hidden
        self.WC = np.random.rand(self.n_hidden, self.n_hidden)  # context -> hidden
        self.W2 = np.random.rand(1, self.n_hidden)  # hidden -> output

    def feed_forward(self, input_bit: int):
        """Feed the given input bit through the network, and
        update the context units.

        Z1 = unactivated hidden layer [2x1]
        A1 = activated hidden layer [2x1]
        Z2 = unactivated output layer [1x1]
        A2 = activated output layer [1x1]
        """
        Z1: Matrix_2X1 = self.W1 @ [[input_bit]] + self.WC @ self.context
        A1: Matrix_2X1 = ElmanRNN.ReLU(Z1)  # hidden layer
        Z2: Matrix_1X1 = self.W2 @ A1
        A2: Matrix_1X1 = ElmanRNN.sigmoid(Z2)
        self.context = A1
        return (Z1, A1, Z2, A2)

    @staticmethod
    def ReLU(Z: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.maximum(0, Z)

    @staticmethod
    def ReLU_derivative(Z):
        return Z > 0

    @staticmethod
    def sigmoid(A2: NDArray[np.floating]) -> NDArray[np.floating]:
        return 1 / (1 + np.exp(-A2))

    @staticmethod
    def get_prediction(A2: Matrix_1X1) -> int:
        """Get predicted bit given output of the network"""
        return 1 if A2[0][0] >= 0.5 else 0

    @staticmethod
    def get_accuracy(predictions: Sequence, sequence: Sequence) -> float:
        """Get the accuracy of the model given predictions on the
        entire dataset.  Since the first bit is not predicted, we
        skip it."""
        if len(predictions) != len(sequence) - 1:
            raise ValueError("Predictions must have the same length as the sequence minus 1")
        return np.mean(predictions == sequence[1:])

    def infer(self, sequence: list[int]) -> int:
        """Infer the output of the network for a given input sequence"""
        self.reset_context()
        for input_bit in sequence:
            _, _, _, A2 = self.feed_forward(input_bit)
        return ElmanRNN.get_prediction(A2)

    def backpropagate(
        self,
        Z1: Matrix_2X1,
        A1: Matrix_2X1,
        A2: Matrix_1X1,
        input_bit: Literal[0, 1],
        target: Literal[0, 1],
    ) -> tuple[Matrix_2X1, Matrix_1X2]:
        """Backpropagate the error through the network; i is the current time step"""
        dL_dZ2: Matrix_1X1 = A2 - [[target]]
        dL_dW2: Matrix_1X2 = dL_dZ2 @ A1.T
        dL_dZ1: Matrix_2X1 = self.W2.T @ dL_dZ2 * ElmanRNN.ReLU_derivative(Z1)
        dL_dW1: Matrix_2X1 = dL_dZ1 @ [[input_bit]]
        return (dL_dW1, dL_dW2)

    def train(self, sequence: Sequence, learning_rate: float, n_passes: int) -> None:
        """Train the model"""
        for i_pass in range(n_passes):
            predictions: Sequence = np.array([])
            for i, input_bit in enumerate(sequence):
                # Skip the last element because we need to predict the next bit
                if i == len(sequence) - 1:
                    break
                # Feed forward
                Z1, A1, _, A2 = self.feed_forward(input_bit)
                predictions = np.append(predictions, ElmanRNN.get_prediction(A2))
                # Backpropagate
                dL_dW1, dL_dW2 = self.backpropagate(Z1, A1, A2, input_bit, sequence[i + 1])
                self.W1 -= learning_rate * dL_dW1
                self.W2 -= learning_rate * dL_dW2
            # Print info
            if i_pass % 1 == 0:
                print(f"Pass {i_pass+1} completed")
                self.print_params_info()
                print(
                    f" - predictions: {predictions.shape}, min: {np.min(predictions)}, max: {np.max(predictions)}, mean: {np.mean(predictions)}"
                )
                print(f" - Accuracy: {ElmanRNN.get_accuracy(predictions, sequence)}")

    def print_params_info(self):
        print(
            f" - W1: {self.W1.shape}, min: {np.min(self.W1)}, max: {np.max(self.W1)}, mean: {np.mean(self.W1)}"
        )
        print(
            f" - W2: {self.W2.shape}, min: {np.min(self.W2)}, max: {np.max(self.W2)}, mean: {np.mean(self.W2)}"
        )
        print(
            f" - WC: {self.WC.shape}, min: {np.min(self.WC)}, max: {np.max(self.WC)}, mean: {np.mean(self.WC)}"
        )


rnn = ElmanRNN(n_units_hidden)
rnn.train(training_sequence, learning_rate, n_passes)
# print(rnn.infer([0, 1, 1]))
# print(rnn.context)
