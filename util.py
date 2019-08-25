from typing import List
from math import exp

# dot product of two vectors

# Explanation:
# dot_product adds up all the products of every pair of numbers included in
# ths list of oredered pairs created mixing two lists. The input data correspond to
# neuron input and weights. Then, the activation function is applied to this result,
# before send the signal to the next layer (see fig 7.2).

def dot_product(xs: List[float], ys: List[float]) -> float:
    return sum(x * y for x, y in zip(xs, ys))


# the classic signoid activation function

# Explanation:
# Referred as 'the sigmoid function', transform (non-linearly) the dot prouct
# and returns always a value between 0 and 1. sigmoid use the Euler constant in its
# equation. Also, we add a derivative version of sigmoid function to using it at 
# backpropagation phase. Both code pieces are programming representation of
# mathematical formulae (see fig 7.7).

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))

def derivative_sigmoid(x :float) -> float:
    sig: float = sigmoid(x)
    return sig * (1 - sig)

# Normalization: We need to normalize ("clean") data before paasing it to network.
# In other words, we need to convert our data so every sample use the same scale.
# This operation is known as feature scaling.
# In this case, every neuron outputs values between 0 and 1 due to the sigmoid
# activation function. It sounds logical that a scale between 0 and 1 would make sense
# for the attributes in our input data set as well. 
# We'll use the following formula: 
# for any value, V, newV = (oldV - min) / (max - min).

# Assumptions:
# - all rows are of equal length
# - feature scale each column to be in the range 0 - 1
# - dataset is a two-dimensional list of floats
def normalize_by_feature_scaling(dataset: List[List[float]]) -> None:
    # The parameter is a reference to the original datasets (not a copy), so modifies it
    for col_num in range(len(dataset[0])):
        column: List[float] = [row[col_num] for row in dataset]
        maximum = max(column)
        minimum = min(column)
        for row_num in range(len(dataset)):
            dataset[row_num][col_num] = (dataset[row_num][col_num] - minimum) / (maximum - minimum)
