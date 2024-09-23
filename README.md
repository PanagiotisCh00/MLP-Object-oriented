# MultiLayer Perceptron (MLP) Implementation
Project Overview

This project implements a Feed Forward Neural Network using object-oriented programming principles in Python. The primary goal is to build a neural network capable of computing the exclusive-OR (XOR) function using the Backpropagation algorithm with momentum. The project is structured to allow reusability in future tasks and experiments.
Features

    Object-Oriented Design: The neural network is implemented using a modular design with distinct classes for neurons, connections, layers, and the overall network.
    Support for Backpropagation with Momentum: The network supports online update training with momentum to enhance the learning process and prevent falling into local minima.
    Multiple Layers: The network includes support for multiple hidden layers, allowing for flexible architecture configurations.
    Training and Testing: The network is trained using the XOR function with custom parameters like learning rate, momentum, and the number of iterations. It also includes functionality for testing the network's performance on test datasets.

How It Works

    Input Parameters: The network configuration (number of neurons per layer, learning rate, momentum, etc.) is defined in a parameters.txt file. Training and testing data are read from training.txt and test.txt files, respectively.
    Network Initialization: The neural network is built with input, hidden, and output layers, each consisting of neurons. Special bias neurons are added for improved accuracy.
    Training: During each training epoch, the network performs:
        Forward Propagation: Calculates the output for each neuron.
        Backward Propagation: Adjusts weights using the calculated errors and momentum.
        Weight Adaptation: Updates the weights based on the learning rate and calculated deltas.
    Error and Accuracy Tracking: The network logs the training and testing error at each iteration, as well as the accuracy. These metrics are saved in errors.txt and successrate.txt files, and visualized as graphs.

Credits

This project was developed as part of the Machine Learning course (EPL442) at the University of Cyprus.
