from data_handling import percepton_data_handling, add_bias
from perceptron import Perceptron
import numpy as np
def main():
    X, y = percepton_data_handling()
    X = add_bias(X)
    model = Perceptron()
    model.fit(X, y)
    model.print_results()

if __name__ == "__main__":
    main()