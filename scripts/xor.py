"""
Elman's simple recurrent neural network (RNN) for the XOR problem.

The script generates a sequence of 3,000 bits where each third bit
is the XOR of the previous two bits.  The network is then trained
on this sequence, bit by bit, using a sigmoid activation and a context
layer.  The predictions are then tested on a new sequence of 90 bits.

usage: python xor.py

The main difference with Elman's original paper is that I use 8 hidden units
instead of 2.  While I also get convergence with 2 hidden neurons, it happens
only with rare random initializations. 8 hidden neurons seem to be more stable.

The parameters I use are:
- 8 hidden units
- learning rate of 0.2
- 50 passes

With these parameters, I achieve 90% accuracy on the training sequence and
90% accuracy on the test sequence.  By accuracy here I mean the accuracy of
predicting the XOR bit (the third bit in the triplet); the accuracy of the
first two bits is always 50% because they are random.

Play with parameters in the first lines of the script to see how they affect
conversion and accuracy.
"""

import numpy as np
from numpy import ndarray, dtype, floating
from typing import Literal
from numpy.typing import NDArray
import matplotlib.pyplot as plt

# Hyperparameters
# Please note that input and output layers both have only 1
# unit (the current and next bit, respectively)
n_passes = 50
n_units_hidden = 8
learning_rate = 0.2

# Dataset parameters
sequence_length = 3000
test_sequence_length = 90

# Set random seed for reproducibility
np.random.seed(42)

# Print nice matrices
np.set_printoptions(precision=2, suppress=True)

# Types
Matrix_2X2 = ndarray[tuple[Literal[2], Literal[2]], dtype[floating]]  # [2x2] matrix
Matrix_2X1 = ndarray[tuple[Literal[2], Literal[1]], dtype[floating]]  # [2x1] column vector
Matrix_1X2 = ndarray[tuple[Literal[1], Literal[2]], dtype[floating]]  # [1x2] row vector
Matrix_1X1 = ndarray[tuple[Literal[1], Literal[1]], dtype[floating]]  # [1x1] scalar
Sequence = NDArray[np.int8]  # sequence of bits


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
        self.b1: Matrix_2X1
        self.b2: Matrix_1X1

        # Initializations
        self.reset_context()
        self.init_params()
        print("Initialiation:")
        self.print_params()

    def loss_cen(self, A2: Matrix_1X1, target: int) -> float:
        """Compute the cross-entropy loss of the network"""
        prediction = A2[0][0]
        return -target * np.log(prediction) - (1 - target) * np.log(1 - prediction)

    def loss_mse(self, A2: Matrix_1X1, target: int) -> float:
        """Compute the mean squared error of the network"""
        return 0.5 * (A2[0][0] - target) ** 2

    def reset_context(self) -> None:
        """Set the context units to 0.5 as per the Elman paper"""
        self.context = np.full((self.n_hidden, 1), 0.5)  # [2x1]

    def init_params(self) -> None:
        """Initialize the model parameters at t0"""
        # Weights
        self.W1 = np.random.uniform(-1, 1, (self.n_hidden, 1))  # input -> hidden
        self.WC = np.random.uniform(-1, 1, (self.n_hidden, self.n_hidden))  # context -> hidden
        self.W2 = np.random.uniform(-1, 1, (1, self.n_hidden))  # hidden -> output
        # Biases
        self.b1 = np.zeros((self.n_hidden, 1))
        self.b2 = np.zeros((1, 1))

    def forward(self, input_bit: int) -> tuple[Matrix_2X1, Matrix_2X1, Matrix_1X1, Matrix_1X1]:
        """Feed the given input bit through the network, and
        update the context units.

        Z1 = unactivated hidden layer [2x1]
        A1 = activated hidden layer [2x1]
        Z2 = unactivated output layer [1x1]
        A2 = activated output layer [1x1]
        """
        Z1: Matrix_2X1 = self.W1 @ [[input_bit]] + self.WC @ self.context + self.b1
        A1: Matrix_2X1 = self.sigmoid(Z1)  # hidden layer
        Z2: Matrix_1X1 = self.W2 @ A1 + self.b2
        A2: Matrix_1X1 = self.sigmoid(Z2)
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
    def sigmoid_derivative(A2: NDArray[np.floating]) -> NDArray[np.floating]:
        return A2 * (1 - A2)

    @staticmethod
    def get_prediction(A2: Matrix_1X1) -> int:
        """Get predicted bit given output of the network"""
        return 1 if A2[0][0] >= 0.5 else 0

    @staticmethod
    def get_accuracy(predictions: Sequence, sequence: Sequence) -> float:
        """Get the accuracy of the model given predictions on the
        entire training sequence."""
        return np.mean(predictions == sequence)

    @staticmethod
    def get_accuracy_by_position(predictions: Sequence, sequence: Sequence) -> list[float]:
        """Get the accuracy of the model given predictions on the
        entire training sequence, split by position in the triplet
        (0,1,2).  We expect the prediction in the third position to
        be more accurate than the first two, because it's the only
        one that is predictable by the network."""
        accuracy = predictions == sequence
        return accuracy.reshape(-1, 3).mean(axis=0)

    @staticmethod
    def generate_xor_sequence(n_triplets: int) -> Sequence:
        """Generate a sequence of bits where each third bit is the XOR of the previous two"""
        pairs = np.random.randint(0, 2, [n_triplets, 2])
        xors = np.logical_xor(pairs[:, 0], pairs[:, 1]).astype(int)
        return np.concatenate([pairs, xors[:, None]], axis=1).flatten()

    def infer(self, sequence: list[int]) -> int:
        """Infer the output of the network for a given input sequence"""
        self.reset_context()
        for input_bit in sequence:
            _, __, ___, A2 = self.forward(input_bit)
        return ElmanRNN.get_prediction(A2)

    def backward(
        self,
        Z1: Matrix_2X1,
        A1: Matrix_2X1,
        A2: Matrix_1X1,
        input_bit: Literal[0, 1],
        target: Literal[0, 1],
    ) -> tuple[Matrix_2X1, Matrix_2X1, Matrix_2X2, Matrix_1X2, Matrix_1X1]:
        """Backpropagate the error through the network; i is the current time step"""
        # With sigmoid + cross-entropy loss + relu
        # dL_dZ2: Matrix_1X1 = A2 - [[target]]
        # dL_dW2: Matrix_1X2 = dL_dZ2 @ A1.T
        # dL_db2: Matrix_1X1 = dL_dZ2
        # dL_dZ1: Matrix_2X1 = self.W2.T @ dL_dZ2 * self.ReLU_derivative(Z1)
        # dL_dW1: Matrix_2X1 = dL_dZ1 @ [[input_bit]]
        # dL_db1: Matrix_2X1 = dL_dZ1
        # dL_dWC: Matrix_2X2 = dL_dZ1 @ self.context.T

        # With sigmoid + mse + sigmoid
        dL_dZ2 = (A2 - [[target]]) * self.sigmoid_derivative(A2)
        dL_dW2 = dL_dZ2 @ A1.T
        dL_db2 = dL_dZ2
        dL_dZ1 = self.W2.T @ dL_dZ2 * self.sigmoid_derivative(A1)
        dL_dW1 = dL_dZ1 @ [[input_bit]]
        dL_db1 = dL_dZ1
        dL_dWC = dL_dZ1 @ self.context.T

        return (dL_dW1, dL_db1, dL_dWC, dL_dW2, dL_db2)

    def train(self, sequence: Sequence, learning_rate: float, n_passes: int) -> None:
        """Train the model"""
        for i_pass in range(n_passes):
            self.reset_context()
            losses_cen = np.array([])
            losses_mse = np.array([])
            predictions: Sequence = np.array(
                [0]
            )  # first prediction is irrelevant because we don't predict the first bit
            for i, input_bit in enumerate(sequence):
                # Skip the last element because we need to predict the next bit
                if i == len(sequence) - 1:
                    break
                # Feed forward
                Z1, A1, _, A2 = self.forward(input_bit)
                losses_cen = np.append(losses_cen, self.loss_cen(A2, sequence[i + 1]))
                losses_mse = np.append(losses_mse, self.loss_mse(A2, sequence[i + 1]))
                predictions = np.append(predictions, ElmanRNN.get_prediction(A2))
                # Backpropagate
                dL_dW1, dL_db1, dL_dWC, dL_dW2, dL_db2 = self.backward(
                    Z1, A1, A2, input_bit, sequence[i + 1]
                )
                self.W1 -= learning_rate * dL_dW1
                self.b1 -= learning_rate * dL_db1
                self.WC -= learning_rate * dL_dWC
                self.W2 -= learning_rate * dL_dW2
                self.b2 -= learning_rate * dL_db2
            # Print info
            if i_pass % 1 == 0:
                print(f"Pass {i_pass+1} completed")
                self.print_params()
                print(f" - Avg cross-entropy loss: {np.sum(losses_cen)/sequence_length}")
                print(f" - Avg Mean squared error: {np.sum(losses_mse)/sequence_length}")
                print(
                    f" - predictions: {predictions.shape}, min: {np.min(predictions)}, max: {np.max(predictions)}, mean: {np.mean(predictions)}"
                )
                print(f" - Accuracy: {ElmanRNN.get_accuracy(predictions, sequence)}")
                self.accpos = ElmanRNN.get_accuracy_by_position(predictions, sequence)
                print(f" - Accuracy in first position: {self.accpos[0]}")
                print(f" - Accuracy in second position: {self.accpos[1]}")
                print(f" - Accuracy in third position: >>> {self.accpos[2]} <<<")

    def print_params(self):
        print("W1:")
        print(self.W1)
        print("WC:")
        print(self.WC)
        print("W2:")
        print(self.W2)


# Create input dataset: 3,000 bits where each third bit is the XOR of
# the previous two bits. For example: 100010011110101...
training_sequence = ElmanRNN.generate_xor_sequence(sequence_length // 3)  # [3000x1]
print(f"Training sequence has shape {training_sequence.shape}:")
print(training_sequence)

# Check that the sequence is correct
print("\nChecking the sequence:")
for i in range(0, len(training_sequence) - 2, 3):
    if training_sequence[i] ^ training_sequence[i + 1] != training_sequence[i + 2]:
        print("Error in sequence")
        break
print("Sequence is correct")

rnn = ElmanRNN(n_units_hidden)
rnn.train(training_sequence, learning_rate, n_passes)

# Print table with predictions on new sequence
print("\nTesting predictions on new sequence...")
test_sequence = ElmanRNN.generate_xor_sequence(test_sequence_length // 3)
rnn.reset_context()

print("\nOriginal sequence vs Predictions:")
print("Pos\tInput\tTarget\tPredicted\tCorrect?")
print("-" * 50)

hits = 0
test_predictions = np.array([])
for i in range(len(test_sequence) - 1):
    _, __, ___, A2 = rnn.forward(test_sequence[i])
    prediction = A2[0][0]
    np.append(test_predictions, prediction)
    pos = i % 3
    if pos == 1:
        if (prediction > 0.5) == test_sequence[i + 1]:
            correct = "Yes"
            hits += 1
        else:
            correct = "No"
        print(f"{pos}\t{test_sequence[i]}\t{test_sequence[i+1]}\t{prediction:.3f}\t{correct}")
    else:
        print(f"{pos}\t{test_sequence[i]}\t{test_sequence[i+1]}\t{prediction:.3f}")

print(f"Correct prediction fro XOR bit: {hits} out of {test_sequence_length // 3} bits")


print(f">>> FINAL ACCURACY RESULTS <<<")
print(f"XOR Accuracy on TRAINING sequence: {rnn.accpos[2]:.2f}")
print(f"XOR Accuracy on TEST sequence: {hits / (test_sequence_length // 3):.2f}")
