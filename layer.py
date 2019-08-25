from __future__ import annotations
from typing import List, Callable, Optional
from random import random
from neuron import Neuron
from util import dot_product

# This class must maintain state elements in an up level.
# It needs to know how many neurons it should be initializing, their
# activation functions and learning rates. In this example, those elements
# will be the same for every neuron.

# Optional[type]: states that the type of the value is either 'type' or None. Is a
# shorthand for Union[type, None] where Union[X, Y] means type "X or Y".

class Layer:
    def __init__(
        self, 
        previous_layer: Optional[Layer], 
        num_neurons: int, 
        learning_rate: float, 
        activation_function: Callable[[float], float],
        derivative_activation_function: Callable[[float], float]
        ) -> None:
        self.previous_layer: Optional[Layer] = previous_layer
        self.neurons: List[Neuron] = []
        # The folowing could all be one large list comprehension
        for _ in range(num_neurons):
            if previous_layer is None:
                random_weights: List[float] = []
            else:
                random_weights = [random() for _ in range(len(previous_layer.neurons))]
            neuron: Neuron = Neuron(random_weights, learning_rate, activation_function, derivative_activation_function)
            self.neurons.append(neuron)
        self.output_cache: List[float] = [0.0 for _ in range(num_neurons)]
    
    # Layer must process every signal through every neuron. outputs returns the result of
    # that processing and caches the output. If there is not previous layer, then this layer is
    # the input layer, so output cache is the input itself.
    # This output cache is different from neuron level output cache. This represent the output
    # of the entire layer, composed by output of every neuron.
    def outputs(self, inputs: List[float]) -> List[float]:
        if self.previous_layer is None:
            self.output_cache = inputs
        else:
            self.output_cache = [n.output(inputs) for n in self.neurons]
        return self.output_cache

    # During backpropagation, we nedd to calculate deltas from the output and hidden layers.
    # As the sigmoid function in Neuron class, these methods are a rote translation of
    # mathematical formulae (see fig 7.4 y 7.5).
    # Note that this method should only be called on otput layer.
    def calculate_deltas_for_output_layer(self, expected: List[float]) -> None:
        for n in range(len(self.neurons)):
            self.neurons[n].delta = self.neurons[n].derivative_activation_function(self.neurons[n].output_cache) * (expected[n] - self.output_cache[n])

    # This method should not be called on output layer
    def calculate_deltas_for_hidden_layer(self, next_layer: Layer) -> None:
        for index, neuron in enumerate(self.neurons):
            next_weights: List[float] = [n.weights[index] for n in next_layer.neurons]
            next_deltas: List[float] = [n.delta for n in next_layer.neurons]
            sum_weights_and_deltas: float = dot_product(next_weights, next_deltas)
            neuron.delta = neuron.derivative_activation_function(neuron.output_cache) * sum_weights_and_deltas
