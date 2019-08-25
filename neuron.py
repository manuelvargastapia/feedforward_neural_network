from typing import List, Callable
from util import dot_product

# Neuron class. Contains all the relevant data for a single neuron, without access
# to other neurons or next organizational level, layers. Programmatically, this class
# only maintains some state data that are used by the other classes.

# Because of output_cache and delta are not known when a neuron is first created,
# these values are initialized to 0.

# Callable: return type referring to a function that, in this case, takes as arguments
# a bunch of float value and return a float.

class Neuron:
    def __init__(
        self, 
        weights: List[float], 
        learning_rate: float, 
        activation_function: Callable[[float], float], 
        derivative_activation_function: Callable[[float], float]
        ) -> None:
        self.weights: List[float] = weights
        self.activation_function: Callable[[float], float] = activation_function
        self.derivative_activation_function: Callable[[float], float] = derivative_activation_function
        self.learning_rate: float = learning_rate
        self.output_cache: float = 0.0
        self.delta: float = 0.0
    
    # Output cache, before passing by activation function, is needed for backpropagation phase
    def output(self, inputs: List[float]) -> float:
        self.output_cache = dot_product(inputs, self.weights)
        return self.activation_function(self.output_cache)
