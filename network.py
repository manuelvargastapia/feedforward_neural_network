from __future__ import annotations
from typing import List, Callable, TypeVar, Tuple
from functools import reduce
from layer import Layer
from util import sigmoid, derivative_sigmoid

# The responsibioty of this class is manage the layers.

# Assumption: all the neurons in the network use the same activation function and learning rate.

# Output type of interpretation of neural network. Its names is "T" and, in this case,
# could be of any type (see docs of TypeVar and Generics).
T = TypeVar('T')

class Network:
    # layer_structure: [input layer, hiden layer, output layer]
    def __init__(
        self, 
        layer_structure: List[int], 
        learning_rate: float,
        activation_function: Callable[[float], float] = sigmoid, 
        derivative_activation_function: Callable[[float], float] = derivative_sigmoid
        ) -> None:
        if len(layer_structure) < 3:
            raise ValueError("Error: should be at least 3 layers (1 input, 1 hidden, 1 output)")
        self.layers: List[Layer] = []
        input_layer: Layer = Layer(None, layer_structure[0], learning_rate, activation_function, derivative_activation_function)
        self.layers.append(input_layer)
        for previous, num_neurons in enumerate(layer_structure[1::]):
            next_layer = Layer(self.layers[previous], num_neurons, learning_rate, activation_function, derivative_activation_function)
            self.layers.append(next_layer)

    # outputs pushes input data to the first layer, then output from the first as output to
    # the second, second to third, etc. It calculate the Network outputs using the Layer 
    # level outputs, that use the Neuron level outputs. reduce is used to compactly pass 
    # signals from one layer to the next recurrently.
    #
    # reduce algorithm:
    # 1) calls functions with two first elements in sequence and returns the value;
    # 2) calls function again with the result from previous call and the next value on
    # sequence;
    # 3) repeat until reach the end of the sequence.
    #
    # Step by step:
    # 1) use input as initial value to get the outputs from first, input, layer;
    # 2) use the returned outputs to get the outputs from the next layer, and so on;
    # 3) finish with the las outputs from the output layer.
    def outputs(self, input: List[float]) -> List[float]:
        return reduce(lambda inputs, layer: layer.outputs(inputs), self.layers, input)

    # backpropagate calculate deltas for every neuron in network. Figure out each 
    # neuron's changes based on the errors of the output versus the expected outcome.
    # This method does not actually change any weight (see update_weights)
    def backpropagate(self, expected: List[float]) -> None:
        # calculate delta for output layer neurons
        last_layer: int = len(self.layers) - 1
        self.layers[last_layer].calculate_deltas_for_output_layer(expected)
        # calculate delta for hidden layers in reverse order (because is a backward process)
        for l in range(last_layer - 1, 0, -1):
            self.layers[1].calculate_deltas_for_hidden_layer(self.layers[l + 1])
    
    # This method must be called after backpropagate, becasuse depends on deltas to
    # modify weights (see formula in fig 7.6)
    def update_weights(self) -> None:
        for layer in self.layers[1:]: # skip input layer
            for neuron in layer.neurons:
                for w in range(len(neuron.weights)):
                    neuron.weights[w] = neuron.weights[w] + (neuron.learning_rate * (layer.previous_layer.output_cache[w]) * neuron.delta)

    # train will call backpropagate with expected values for a given input, then update
    # the weights according to deltas.
    def train(self, inputs: List[List[float]], expecteds: List[List[float]]) -> None:
        for location, xs in enumerate(inputs): # get index and value
            ys: List[float] = expecteds[location]
            outs: List[float] = self.outputs(xs)
            self.backpropagate(ys)
            self.update_weights()

    # For generalized results that require classification
    # this function will return the correct number of trials
    # and the percentage correct out of the total
    def validate(
        self, 
        inputs: List[List[float]], 
        expecteds: List[T], 
        interpret_output: Callable[[List[float]], T]
        ) -> Tuple[int, int, float]: 
        correct: int = 0
        for input, expected in zip(inputs, expecteds):
            result: T = interpret_output(self.outputs(input))
            if result == expected:
                correct += 1
        percentage: float = correct / len(inputs)
        return correct, len(inputs), percentage
        