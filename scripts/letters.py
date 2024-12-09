"""
Elman's simple recurrent neural network (RNN) to predict the next letter.

The model correctly predicts the next vowel with high accuracy, and whether
a consonant is coming up next, as described in the "STRUCTURE IN LETTER SEQUENCES"
section of Elman's 1990 paper.

The model will be tested against a new sequence of 1,000 letters, and the accuracy
printed to screen. With a random seed of 42, the model attains the following
accuracy:

- VOWELS accuracy: 95% (should tend to 100%)
- CONSONANTS accuracy: 31% (should tend to 33.3%)
- CONSONANT COMING UP accuracy: 94% (should tend to 100%)

usage: python letters.py

Play with parameters in the first lines of the script to see how they affect
conversion and accuracy.  You can also decide to encode the letters in one-hot
encoding instead of the encoding defined in Elman's paper.
"""

import numpy as np
from numpy import ndarray, dtype, floating
from typing import Literal
from numpy.typing import NDArray

# Hyperparameters
# Please note that input and output layers both have 6 unit
n_passes = 50  # Elman uses 200 passes
n_units_hidden = 20
learning_rate = 0.2

# Dataset parameters
n_consonants = 1000
n_consonants_test = 1000
one_hot = False

# Set random seed for reproducibility
np.random.seed(42)

# Print nice matrices
np.set_printoptions(precision=2, suppress=True)

# Types
Matrix_20X20 = ndarray[tuple[Literal[20], Literal[20]], dtype[floating]]  # [20x20] matrix
Matrix_20X6 = ndarray[tuple[Literal[20], Literal[6]], dtype[floating]]  # [20x6] matrix
Matrix_6X20 = ndarray[tuple[Literal[6], Literal[20]], dtype[floating]]  # [6x20] matrix
Matrix_20X1 = ndarray[tuple[Literal[20], Literal[1]], dtype[floating]]  # [20x1] column vector
Matrix_6X1 = ndarray[tuple[Literal[6], Literal[1]], dtype[floating]]  # [6x1] column vector
Letter = Literal["b", "d", "g", "a", "i", "u"]
SequenceOfLetters = list[Letter]
SequenceEncoded = list[Matrix_6X1]


# It is best to represent the NN with a class because we need to keep
# track of the timesteps and of the context units
class ElmanRNN:

    def __init__(self, n_hidden: int, one_hot: bool) -> None:
        # Assignments
        self.n_hidden = n_hidden
        self.one_hot = one_hot

        # Types
        self.context: Matrix_20X1
        self.W1: Matrix_20X6
        self.WC: Matrix_20X20
        self.W2: Matrix_6X20
        self.b1: Matrix_20X1
        self.b2: Matrix_6X1

        # Initializations
        self.reset_context()
        self.init_params()
        print("Initialiation:")
        self.print_params()

    def loss_cen(self, A2: Matrix_6X1, target: Matrix_6X1) -> float:
        """Compute the cross-entropy loss of the network"""
        return -np.sum(target * np.log(A2))

    def loss_mse(self, A2: Matrix_6X1, target: Matrix_6X1) -> float:
        """Compute the mean squared error of the network"""
        return 0.5 * sum((A2 - target) ** 2)

    def reset_context(self) -> None:
        """Set the context units to 0.5 as per the Elman paper"""
        self.context = np.full((self.n_hidden, 1), 0.5)  # [20x1]

    def init_params(self) -> None:
        """Initialize the model parameters at t0"""
        # Weights
        self.W1 = np.random.uniform(-1, 1, (self.n_hidden, 6))  # input -> hidden
        self.WC = np.random.uniform(-1, 1, (self.n_hidden, self.n_hidden))  # context -> hidden
        self.W2 = np.random.uniform(-1, 1, (6, self.n_hidden))  # hidden -> output
        # Biases
        self.b1 = np.zeros((self.n_hidden, 1))
        self.b2 = np.zeros((6, 1))

    def forward(
        self, encoded_letter: Matrix_6X1
    ) -> tuple[Matrix_20X1, Matrix_20X1, Matrix_6X1, Matrix_6X1]:
        """Feed the given input letter, in encoded form, through the network, and
        update the context units.

        Z1 = unactivated hidden layer [20x1]
        A1 = activated hidden layer [20x1]
        Z2 = unactivated output layer [6x1]
        A2 = activated output layer [6x1]
        """
        Z1: Matrix_20X1 = self.W1 @ encoded_letter + self.WC @ self.context + self.b1
        A1: Matrix_20X1 = self.sigmoid(Z1)  # hidden layer
        Z2: Matrix_6X1 = self.W2 @ A1 + self.b2
        A2: Matrix_6X1 = self.sigmoid(Z2)
        self.context = A1
        return (Z1, A1, Z2, A2)

    @staticmethod
    def sigmoid(A2: NDArray[np.floating]) -> NDArray[np.floating]:
        return 1 / (1 + np.exp(-A2))

    @staticmethod
    def sigmoid_derivative(A2: NDArray[np.floating]) -> NDArray[np.floating]:
        return A2 * (1 - A2)

    def get_predicted_letter(self, A2: Matrix_6X1) -> Letter:
        """Get predicted letter given output of the network.  This is different
        than the raw prediction which is a 6 element vector."""
        return self.closest_letter(A2)

    @staticmethod
    def get_accuracy(predictions: SequenceOfLetters, sequence: SequenceOfLetters) -> float:
        """Get the accuracy of the model given predictions on the
        entire training sequence; this is the percentage of correct
        predictions of actual letters, not vectors."""
        return float(np.mean(np.array(predictions) == np.array(sequence)))

    def get_asymptotic_accuracy(self, sequence: SequenceOfLetters) -> float:
        """Get the expected accuracy, assuming that all vowels can be predicted
        and that there's a 33% chance to predict a consonants."""
        n_sequence = len(sequence)
        n_vowels = len(self.get_vowels_in_sequence(sequence))
        n_consonants = len(self.get_consonants_in_sequence(sequence))
        return n_vowels / n_sequence + n_consonants / n_sequence * 1 / 3

    @staticmethod
    def generate_letter_sequence(n_consonants: int) -> SequenceOfLetters:
        """Generate a sequence of letters according to Elman's rule:
        First, the three consonants (b, d, g) were combined in random order to
        obtain a 1,000-letter sequence. Then, each consonant was replaced using
        the rules b -> ba, d -> dii, g -> guuu."""
        consonants_sequence = np.random.choice(["b", "d", "g"], n_consonants)
        letters_sequence: SequenceOfLetters = []
        for consonant in consonants_sequence:
            if consonant == "b":
                letters_sequence += ["b", "a"]
            elif consonant == "d":
                letters_sequence += ["d", "i", "i"]
            elif consonant == "g":
                letters_sequence += ["g", "u", "u", "u"]
        return letters_sequence

    @staticmethod
    def get_vowels_in_sequence(sequence: SequenceOfLetters) -> SequenceOfLetters:
        """Return the list of vowels in the sequence"""
        return [letter for letter in sequence if letter in ["a", "i", "u"]]

    @staticmethod
    def get_consonants_in_sequence(sequence: SequenceOfLetters) -> SequenceOfLetters:
        """Count the list of consonants in the sequence"""
        return [letter for letter in sequence if letter in ["b", "d", "g"]]

    def encode_letter(self, letter: Letter) -> Matrix_6X1:
        """Encode a single letter into a 6x1 vector as defined in table 1
        of Elman's paper; optionally do one hot encoding instead"""
        if self.one_hot:
            # One-hot encode to a 6x1 vector, in the order b,d,g,a,i,u
            encoding = {
                "b": np.array([[1, 0, 0, 0, 0, 0]]).T,
                "d": np.array([[0, 1, 0, 0, 0, 0]]).T,
                "g": np.array([[0, 0, 1, 0, 0, 0]]).T,
                "a": np.array([[0, 0, 0, 1, 0, 0]]).T,
                "i": np.array([[0, 0, 0, 0, 1, 0]]).T,
                "u": np.array([[0, 0, 0, 0, 0, 1]]).T,
            }
        else:
            # Encode to a 6x1 vector as defined in table 1 of Elman's paper
            # following phonological rules
            encoding = {
                "b": np.array([[1, 0, 1, 0, 0, 1]]).T,
                "d": np.array([[1, 0, 1, 1, 0, 1]]).T,
                "g": np.array([[1, 0, 1, 0, 1, 1]]).T,
                "a": np.array([[0, 1, 0, 0, 1, 1]]).T,
                "i": np.array([[0, 1, 0, 1, 0, 1]]).T,
                "u": np.array([[0, 1, 0, 1, 1, 1]]).T,
            }
        return encoding[letter]

    def closest_letter(self, A2: Matrix_6X1) -> Letter:
        """Get the closest letter to the output of the network"""
        distances: dict[Letter, float] = {
            "b": float(np.linalg.norm(A2 - self.encode_letter("b"))),
            "d": float(np.linalg.norm(A2 - self.encode_letter("d"))),
            "g": float(np.linalg.norm(A2 - self.encode_letter("g"))),
            "a": float(np.linalg.norm(A2 - self.encode_letter("a"))),
            "i": float(np.linalg.norm(A2 - self.encode_letter("i"))),
            "u": float(np.linalg.norm(A2 - self.encode_letter("u"))),
        }
        return min(distances, key=distances.get)

    def infer(self, sequence: SequenceOfLetters) -> Letter:
        """Infer the output of the network for a given input sequence of letters"""
        self.reset_context()
        for letter in sequence:
            _, __, ___, A2 = self.forward(self.encode_letter(letter))
        return self.get_predicted_letter(A2)

    def backward(
        self,
        Z1: Matrix_20X1,
        A1: Matrix_20X1,
        A2: Matrix_6X1,
        input_letter_encoded: Matrix_6X1,
        target_letter_encoded: Matrix_6X1,
    ) -> tuple[Matrix_20X6, Matrix_20X1, Matrix_20X20, Matrix_6X20, Matrix_6X1]:
        """Backpropagate the error through the network; i is the current time step"""
        # With sigmoid + mse + sigmoid
        dL_dZ2 = (A2 - target_letter_encoded) * self.sigmoid_derivative(A2)
        dL_dW2 = dL_dZ2 @ A1.T
        dL_db2 = dL_dZ2
        dL_dZ1 = self.W2.T @ dL_dZ2 * self.sigmoid_derivative(A1)
        dL_dW1 = dL_dZ1 @ input_letter_encoded.T
        dL_db1 = dL_dZ1
        dL_dWC = dL_dZ1 @ self.context.T

        return (dL_dW1, dL_db1, dL_dWC, dL_dW2, dL_db2)

    def train(self, sequence: SequenceOfLetters, learning_rate: float, n_passes: int) -> None:
        """Train the model"""
        for i_pass in range(n_passes):
            self.reset_context()
            losses_cen = np.array([])
            losses_mse = np.array([])
            # Initialize predictions lists; the first prediction is arbitrary
            # because we cannot predict the first element of the sequence
            raw_predictions = self.encode_letter("b")
            predictions: SequenceOfLetters = ["b"]
            for i, input_letter in enumerate(sequence):
                # Skip the last element because we need to predict the next letter
                if i == len(sequence) - 1:
                    break
                # Target letter in letter space and encoded space
                target_letter = sequence[i + 1]
                target_letter_encoded = self.encode_letter(target_letter)
                # Feed forward
                Z1, A1, _, A2 = self.forward(self.encode_letter(input_letter))
                losses_cen = np.append(losses_cen, self.loss_cen(A2, target_letter_encoded))
                losses_mse = np.append(losses_mse, self.loss_mse(A2, target_letter_encoded))
                raw_predictions = np.column_stack((raw_predictions, A2))
                predictions.append(self.get_predicted_letter(A2))
                # Backpropagate
                dL_dW1, dL_db1, dL_dWC, dL_dW2, dL_db2 = self.backward(
                    Z1, A1, A2, self.encode_letter(input_letter), target_letter_encoded
                )
                self.W1 -= learning_rate * dL_dW1
                self.b1 -= learning_rate * dL_db1
                self.WC -= learning_rate * dL_dWC
                self.W2 -= learning_rate * dL_dW2
                self.b2 -= learning_rate * dL_db2
            # Print info
            if i_pass % 1 == 0:
                print(f"Pass {i_pass+1} completed")
                # self.print_params()
                print(f" - Avg cross-entropy loss: {np.sum(losses_cen)/len(sequence)}")
                print(f" - Avg Mean squared error: {np.sum(losses_mse)/len(sequence)}")
                print(
                    f" - Raw predictions: {raw_predictions.shape}, min: {np.min(raw_predictions)}, max: {np.max(raw_predictions)}, mean: {np.mean(raw_predictions)}"
                )
                print(f" - Accuracy: {rnn.get_accuracy(predictions, sequence)}")
                print(f" - Asymptotic accuracy: {rnn.get_asymptotic_accuracy(sequence)}")
                # self.accpos = ElmanRNN.get_accuracy_by_position(predictions, sequence)
                # print(f" - Accuracy in first position: {self.accpos[0]}")
                # print(f" - Accuracy in second position: {self.accpos[1]}")
                # print(f" - Accuracy in third position: >>> {self.accpos[2]} <<<")

    def print_params(self):
        print("W1:")
        print(self.W1)
        print("WC:")
        print(self.WC)
        print("W2:")
        print(self.W2)


# Instantiate rnn object
rnn = ElmanRNN(n_units_hidden, one_hot=one_hot)

# Create input dataset: 1,000 random consonants each followed by a variable
# number of vowels. For example: diibaguuubadiidiiguuu
training_sequence = rnn.generate_letter_sequence(n_consonants)  # about 3000 letters
print(f"Training sequence has {len(training_sequence)} elements:")

# Check that the sequence is correct
print("\nChecking the sequence:")
for letter in training_sequence:
    if letter not in ["b", "d", "g", "a", "i", "u"]:
        print("Error in sequence")
        break
print("Sequence is correct")

# Train the network
rnn.train(training_sequence, learning_rate, n_passes)

# Print table with predictions on new sequence
print(f"\nTesting predictions on new sequence with {n_consonants_test} letters...")
test_sequence = rnn.generate_letter_sequence(n_consonants_test)
rnn.reset_context()

print("\nOriginal sequence vs Predictions:")
print("Class\tInput\tTarget\tRaw Predicted\tPredicted\tCorrect?")
print("-" * 50)

# Elman in his paper highlights three interesting behaviours:
# 1. The network can predict the next vowel with high accuracy, because
#    the vowels always appear in the same order after a given consonant.
n_vowels = 0
hits_vowels = 0
# 2. The network cannot predict which consonant is next, because they
#    are randomly assigned in the training sequence.
n_consonants = 0
hits_consonants = 0
# 3. The network can predict when a consonant is coming next, because
#    a consonants always follows the last vowel in a sequence of vowels.
#    This prediction is encoded in the first bit of the output vector,
#    that is, the one bit that is turned on only for consonants (see
#    `encode_letter` method).
hits_consonant_coming_up = 0
# 4. As a corollary of the above, the change to get a consonant right
#    is 1/3, because we always know when a consonant is coming up, but
#    we don't know which one, and there are three consonants.

case: Literal["V", "C"]

for i, input_letter in enumerate(test_sequence):
    if i == len(test_sequence) - 1:
        break
    target_letter = test_sequence[i + 1]
    _, __, ___, A2 = rnn.forward(rnn.encode_letter(test_sequence[i]))
    prediction = rnn.get_predicted_letter(A2)
    # Consonants are not preidctable
    if target_letter in ["b", "d", "g"]:
        case = "C"
        n_consonants += 1
        if prediction == target_letter:
            hits_consonants += 1
        if input_letter not in ["a", "i", "u"]:
            raise ValueError("Error in sequence, consonant not following a vowel")
        if prediction in ["b", "d", "g"]:
            # Fun fact: you get roughly the same result if you check the consonant bit:
            # if A2[0] > 0.5:
            # (but only if you are not using one-hot encoding)
            hits_consonant_coming_up += 1
    # Vowels are predictable
    elif target_letter in ["a", "i", "u"]:
        case = "V"
        n_vowels += 1
        if prediction == target_letter:
            hits_vowels += 1
    else:
        print(f"Error in sequence, letter {input_letter} not recognized")
        break
    print(f"{case}\t{input_letter}\t{target_letter}\t{A2.flatten()}\t{prediction}")


print(
    f"VOWELS accuracy: {hits_vowels} out of {n_vowels} [{hits_vowels/n_vowels*100:.2f}%] > SHOULD TEND TO 100%]"
)
print(
    f"CONSONANTS accuracy: {hits_consonants} out of {n_consonants} [{hits_consonants/n_consonants*100:.2f}% > SHOULD TEND TO 33.3%]"
)
print(
    f"CONSONANTS COMING UP accuracy: {hits_consonant_coming_up} out of {n_consonants} [{hits_consonant_coming_up/n_consonants*100:.2f}% > SHOULD TEND TO 100%]"
)
