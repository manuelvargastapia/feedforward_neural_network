# This is a test for a classfication problem that will be solved by our neural network.
# The dataset includes 150 data points, divided in three species of iris flowers.
# Example: 5.1,3.5,1.4,0.2,Iris-setosa. Where every number is an attribute (sepal length, 
# sepal width, petal length, and petal width, respectively).

import csv
from typing import List
from util import normalize_by_feature_scaling
from network import Network
from random import shuffle

if __name__ == "__main__":
    iris_parameters: List[List[float]] = [] # Collection of four attr per sample
    iris_classifications: List[List[float]] = [] # Actual classification of each sample
    iris_species: List[str] = []
    with open('iris.csv', mode='r') as iris_file:
        irises: List = list(csv.reader(iris_file))
        shuffle(irises) # get our lines of data in random order
        for iris in irises:
            parameters: List[float] = [float(n) for n in iris[0:4]]
            iris_parameters.append(parameters)
            species: str = iris[4]
            # Our neural network will have three output neurons, with each representing
            # one possible species. For instance, a final set of outputs of [0.9, 0.3, 0.1] 
            # will represent a classification of iris-setosa, because the first neuron 
            # represents that species, and it is the largest number.
            if species == "Iris-setosa":
                iris_classifications.append([1.0, 0.0, 0.0])
            elif species == "Iris-versicolor":
                iris_classifications.append([0.0, 1.0, 0.0])
            else:
                iris_classifications.append([0.0, 0.0, 1.0])
            iris_species.append(species)
    normalize_by_feature_scaling(iris_parameters)

# Defining neural network

# This network have 4 input neurons (for every attr), 6 for hidden layer (trial and error decision) 
# and 3 output neurons (for the three species). The second argument is the learning rate
# (alse defined by trial and error)
iris_network: Network = Network([4, 6, 3], 0.3)

# Utility function that will be passed to the network’s
# validate() method to help identify correct classifications.
def iris_interpret_output(output: List[float]) -> str:
    if max(output) == output[0]:
        return "Iris-setosa"
    elif max(output) == output[1]:
        return "Iris-versicolor"
    else:
        return "Iris-virginica"

# Training neural network

# Train over the first 140 irises in the data set 50 times
iris_trainers: List[List[float]] = iris_parameters[0:140]
iris_trainers_corrects: List[List[float]] = iris_classifications[0:140]
for _ in range(50):
    iris_network.train(iris_trainers, iris_trainers_corrects)

# Testing neural netowk

# Test over the last 10 of the irises in the data set
iris_testers: List[List[float]] = iris_parameters[140:150]
iris_testers_corrects: List[str] = iris_species[140:150]
iris_results = iris_network.validate(iris_testers, iris_testers_corrects, iris_interpret_output)
print(f"{iris_results[0]} correct of {iris_results[1]} = {iris_results[2] * 100}%")
