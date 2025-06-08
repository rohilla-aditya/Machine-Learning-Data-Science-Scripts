#!/usr/bin/env python
# coding: utf-8

# In[2]:


# %load gradient_descent_techniques super final.py
import numpy as np
import math
from load_fashion_mnist import mnist
import matplotlib.pyplot as plt
import pdb
import sys, ast
import datetime
import time

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = {}
    cache["Z"] = Z
    return A, cache

def sigmoid_der(dA, cache):
    A, _ = sigmoid(cache["Z"])
    dZ = dA * A * (1-A)
    
    return dZ

def relu(Z):
    A = np.maximum(0,Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def relu_der(dA, cache):
    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    dZ[Z<0] = 0
    return dZ

def linear(Z):
    '''
    computes linear activation of Z
    This function is implemented for completeness

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = Z
    cache = {}
    cache["Z"] = Z
    return A, cache

def linear_der(dA, cache):
    '''
    computes derivative of linear activation
    This function is implemented for completeness

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    return dZ

def softmax_cross_entropy_loss(Z, Y=np.array([])):
    '''
    Computes the softmax activation of the inputs Z
    Estimates the cross entropy loss

    Inputs: 
        Z - numpy.ndarray (n, m) (10, 6000)
        Y - numpy.ndarray (1, m) of labels
            when y=[] loss is set to []
    
    Returns:
        A - numpy.ndarray (n, m) of softmax activations
        cache -  a dictionary to store the activations later used to estimate derivatives
        loss - cost of prediction
    '''

    A = ([np.exp(z-np.max(z)) for z in Z.T]/np.sum([np.exp(z-np.max(z)) for z in Z.T], axis=1, keepdims=1)).T

    cache = {}
    cache["A"] = A
    
    loss = 0
    if Y.size > 0:
        n, m = np.shape(Z)
        labels = np.zeros((m, n))
        labels[np.arange(m), Y.astype(int)] = 1
        labels = labels.T #(n,m) (10, 6000)
        epsilon = 0.00001
        loss = -1* np.mean([np.sum([labels[r,c]*math.log(A[r,c] + epsilon) for r in range(n)]) for c in range(m)])
    ###
    
    return A, cache, loss

def softmax_cross_entropy_loss_der(Y, A):
    '''
    Computes the derivative of softmax activation and cross entropy loss

    Inputs: 
        Y - numpy.ndarray (1, m) of labels
        cache -  a dictionary with cached activations A of size (n,m)

    Returns:
        dZ - numpy.ndarray (n, m) derivative for the previous layer
    '''

    n, m = np.shape(A)
    labels = np.zeros((m, n))
    labels[np.arange(m), Y.astype(int)] = 1
    labels = labels.T #(n,m) (10, 6000)

    # number of training examples
    m = A.shape[1]
    dZ = A - labels
    dZ = dZ / m
    ###

    return dZ

def initialize_multilayer_weights(net_dims):
    '''
    Initializes the weights of the multilayer network

    Inputs: 
        net_dims - tuple of network dimensions

    Returns:
        dictionary of parameters
    '''
    print(net_dims)
    np.random.seed(0)
    numLayers = len(net_dims)
    parameters = {}
    velocity = {}
    stored_grads = {}
    for l in range(numLayers-1):
        ###
        parameters["W"+str(l+1)] = np.random.randn(net_dims[l+1], net_dims[l])*0.01
        parameters["b"+str(l+1)] = np.random.randn(net_dims[l+1], 1)*0.01
        velocity["Vw"+str(l+1)] = np.zeros((net_dims[l+1], net_dims[l]))
        velocity["Vb"+str(l+1)] = np.zeros((net_dims[l+1], 1))
        stored_grads["dw"+str(l+1)] = np.zeros((net_dims[l+1], net_dims[l]))
        stored_grads["db"+str(l+1)] = np.zeros((net_dims[l+1], 1))
        ###
    return parameters, velocity, stored_grads

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    
    cache = {}
    cache["A"] = A
    return Z, cache

def layer_forward(A_prev, W, b, activation):
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "sigmoid":
        A, act_cache = sigmoid(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)
    
    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    return A, cache

def multi_layer_forward(X, parameters):
    L = len(parameters)//2  
    A = X
    caches = []
    for l in range(1,L):  # since there is no W0 and b0
        A, cache = layer_forward(A, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches.append(cache)
    
    AL, cache = layer_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "linear")
    caches.append(cache)
    return AL, caches

def linear_backward(dZ, cache, W, b):
    A_prev = cache["A"]
    dA_prev = np.dot(W.T, dZ)
    db = np.sum(dZ, axis=1, keepdims=1)
    dW = np.dot(dZ, cache["A"].T)

    return dA_prev, dW, db

def layer_backward(dA, cache, W, b, activation):
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    m = np.shape(dA)[1]
    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)
    elif activation == "relu":
        dZ = relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = linear_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    dW = dW / m
    db = db / m
    return dA_prev, dW, db

def multi_layer_backward(dAL, caches, parameters):
    L = len(caches)  # with one hidden layer, L = 2
    gradients = {}
    dA = dAL
    activation = "linear"
    for l in reversed(range(1,L+1)):
        dA, gradients["dW"+str(l)], gradients["db"+str(l)] =                     layer_backward(dA, caches[l-1],                     parameters["W"+str(l)],parameters["b"+str(l)],                    activation)
        activation = "relu"
    return gradients

def classicalM(velocity, gradients, beta):
    L = len(gradients)//2
    for l in range(L):
        velocity["Vw"+str(l+1)] = beta * velocity["Vw"+str(l+1)] + gradients["dW"+str(l+1)]
        velocity["Vb"+str(l+1)] = beta * velocity["Vb"+str(l+1)] + gradients["db"+str(l+1)]
    return velocity

def NAG_momentum(velocity, gradients, beta):
    L = len(gradients)//2
    for l in range(L):
        velocity["Vw"+str(l+1)] = beta * velocity["Vw"+str(l+1)] - gradients["dW"+str(l+1)]
        velocity["Vb"+str(l+1)] = beta * velocity["Vb"+str(l+1)] - gradients["db"+str(l+1)]
    return velocity

def NAG(X, Y, parameters, velocity, beta):
    parameters_forw_estimate=dict(parameters)
    L = len(parameters)//2
    for l in range(L):
        parameters["W"+str(l+1)] += (beta * velocity["Vw"+str(l+1)])
        parameters["b"+str(l+1)] += (beta * velocity["Vb"+str(l+1)])
    
    ## call to multi_layer_forward to get activations
    VAL, Vcaches = multi_layer_forward(X, parameters_forw_estimate)
    ## call to softmax cross entropy loss
    VA, Vcache, Vcost = softmax_cross_entropy_loss(VAL, Y)
    ## call to softmax cross entropy loss der
    VdAL = softmax_cross_entropy_loss_der(Y, VA)
    ## call to multi_layer_backward to get gradients
    gradients = multi_layer_backward(VdAL, Vcaches, parameters_forw_estimate)
    ## update velocities
    velocity = NAG_momentum(velocity, gradients, beta)

    return velocity

def update_rmsprop(parameters, stored_grads, epoch, learning_rate, gradients, decay_rate=0.0):
    alpha = learning_rate * 1
    e = 1e-8
    L = len(parameters) // 2

    for l in range(L):
        parameters["W"+str(l+1)] -= alpha * (gradients["dW"+str(l+1)] / np.sqrt(stored_grads["dw"+str(l+1)] + e))
        parameters["b"+str(l+1)] -= alpha * (gradients["db"+str(l+1)] / np.sqrt(stored_grads["db"+str(l+1)] + e))

    return parameters, alpha

def update_adam(parameters, velocity, learning_rate, stored_grads, epoch, beta, decay_rate=0.0):
    alpha = learning_rate*(1/(1+decay_rate*epoch))
    e = 1e-8
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] -= alpha * (velocity["Vw" + str(l+1)] / np.sqrt(stored_grads["dw" + str(l+1)] + e))
        parameters["b" + str(l+1)] -= alpha * (velocity["Vb" + str(l+1)] / np.sqrt(stored_grads["db" + str(l+1)] + e))

    return parameters, alpha


def rmsprop(stored_grads, gradients, beta):
    L = len(gradients) // 2

    for l in range(L):
        stored_grads["dw" + str(l+1)] = beta * stored_grads["dw" + str(l+1)] + (gradients["dW" + str(l+1)] * gradients["dW" + str(l+1)])
        stored_grads["db" + str(l+1)] = beta * stored_grads["db" + str(l+1)] + (gradients["db" + str(l+1)] * gradients["db" + str(l+1)])
    
    return stored_grads



def classify(X, parameters):
    # Forward propagate X using multi_layer_forward
    AL, caches = multi_layer_forward(X, parameters)

    # Get predictions using softmax_cross_entropy_loss
    activations, _, _ = softmax_cross_entropy_loss(AL)

    # Estimate the class labels using predictions
    Ypred = np.argmax(activations, axis=0)

    return Ypred


def update_parameters(parameters, gradients, velocity, epoch, learning_rate, decay_rate=0.0):
    alpha = learning_rate*(1/(1+decay_rate*epoch))
    L = len(parameters)//2
    for l in range(L):
        gradients["dW"+str(l+1)] = velocity["Vw"+str(l+1)]
        gradients["db"+str(l+1)] = velocity["Vb"+str(l+1)]

    for l in range(L):
        parameters["W"+str(l+1)] -= (alpha * gradients["dW"+str(l+1)])
        parameters["b"+str(l+1)] -= (alpha * gradients["db"+str(l+1)])
    
    return parameters, alpha

def update_parameters_no_momentum(parameters, gradients, epoch, learning_rate, decay_rate=0.0):
    '''
        Updates the network parameters with gradient descent
        
        Inputs:
        parameters - dictionary of network parameters
        {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
        gradients - dictionary of gradient of network parameters
        {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
        epoch - epoch number
        learning_rate - step size for learning
        decay_rate - rate of decay of step size - not necessary - in case you want to use
        '''
    alpha = learning_rate*(1/(1+decay_rate*epoch))
    L = len(parameters)//2
    ### CODE HERE
    for l in range(L):
        parameters["W"+str(l+1)] -= (alpha * gradients["dW"+str(l+1)])
        parameters["b"+str(l+1)] -= (alpha * gradients["db"+str(l+1)])
    ###
    
    return parameters, alpha

def multi_layer_network(X, Y, X_val, Y_val, net_dims, technique, num_iterations, learning_rate, batch_size, decay_rate=0.00, beta=0.9):
    parameters, velocity, stored_grads = initialize_multilayer_weights(net_dims)
    A0 = X
    costs = []
    costs_val = []
    m = 0
    v = 0
    t = 0
    alpha = learning_rate
    

    for ii in range(num_iterations):
        # Forward Prop
        AL, caches = multi_layer_forward(X, parameters)
        #AL_val, caches_val = multi_layer_forward(X_val, parameters)
        
        ## call to softmax cross entropy loss
        A, cache, cost = softmax_cross_entropy_loss(AL, Y)
        #_, _, cost_val = softmax_cross_entropy_loss(AL_val, Y_val)
        
        # Backward Prop
        dAL = softmax_cross_entropy_loss_der(Y, A)
        gradients = multi_layer_backward(dAL, caches, parameters)
        
        if technique == 'cm':
            velocity = classicalM(velocity, gradients, beta)
            parameters, alpha = update_parameters(parameters, gradients, velocity, ii, learning_rate, decay_rate=0.01)
        elif technique == 'nag':
            beta = 0.5
            velocity = NAG(A0, Y, parameters, velocity, beta)
            parameters, alpha = update_parameters(parameters, gradients, velocity, ii, learning_rate, decay_rate=0.01)
        elif technique == 'rms':
            stored_grads = rmsprop(stored_grads, gradients, beta)
            parameters, alpha = update_rmsprop(parameters, stored_grads, ii, learning_rate, gradients, decay_rate=0.0)
        elif technique == 'adam':
            velocity = classicalM(velocity, gradients, beta)
            stored_grads = rmsprop(stored_grads, gradients, beta)
            parameters, alpha = update_adam(parameters, velocity, learning_rate, stored_grads, ii, beta)
        else:
            parameters, alpha = update_parameters_no_momentum(parameters, gradients, ii, learning_rate, decay_rate=0.01)
 
        #velocity = classicalM(velocity, gradients, beta)
        #stored_grads = rmsprop(stored_grads, gradients, beta)
        #velocity = NAG(A0, Y, parameters, velocity, beta)

        ## call to update the parameters
        #parameters, alpha = update_adam(parameters, velocity, alpha, stored_grads, ii, beta)
        #parameters, alpha = update_parameters(parameters, gradients, velocity, ii, learning_rate, decay_rate=0.01)
        #parameters, alpha = update_rmsprop(parameters, stored_grads, ii, learning_rate, gradients, decay_rate=0.0)
        #parameters, alpha = update_parameters_no_momentum(parameters, gradients, ii, learning_rate, decay_rate=0.0):

        if ii % 10 == 0:
            costs.append(cost)
        #costs_val.append(cost_val)
        if ii % 10 == 0:
            print("Cost at iteration %i is: %.05f, learning rate: %.05f" %(ii, cost, alpha))
    
    #return costs, costs_val, parameters
    return costs, parameters

def main():
    net_dims = [784 ,500 ,100 ,10]
    #net_dims = ast.literal_eval( sys.argv[1] )
    #net_dims.append(10) # Adding the digits layer with dimensionality = 10
    print("Network dimensions are:" + str(net_dims))

    # getting the subset dataset from MNIST
    train_data, train_label, test_data, test_label = mnist(noTrSamples=6000,noTsSamples=1000,            digit_range=[0,1,2,3,4,5,6,7,8,9],            noTrPerClass=600, noTsPerClass=100)

    # initialize learning rate and num_iterations
    learning_rate = 0.001
    num_iterations = 500
    batch_size = 1500

    iterations = [x*10 for x in range(50)]
    
    #1
    print("NAG")
    a = time.time()
    costs1, parameters = multi_layer_network(train_data, train_label, test_data, test_label, net_dims, 'nag', num_iterations=num_iterations, learning_rate=learning_rate, batch_size=batch_size)
    plt.plot(iterations, costs1, color='#FF7F32')
    b = time.time()
    print('Time taken by algorithm ' + str(b - a) + ' seconds')

    train_Pred = classify(train_data, parameters)
    print(train_Pred)
    test_Pred = classify(test_data, parameters)

    trAcc = np.mean(np.equal(train_Pred, train_label).astype(int))*100
    teAcc = np.mean(np.equal(test_Pred, test_label).astype(int))*100
    print("Accuracy for training set is {0:0.3f} %".format(trAcc))
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))
    
    #2
    print("CM")
    a = time.time()
    costs2, parameters = multi_layer_network(train_data, train_label, test_data, test_label, net_dims, 'cm', num_iterations=num_iterations, learning_rate=learning_rate, batch_size=batch_size)
    plt.plot(iterations, costs2, color='#00A3E0')
    b = time.time()
    print('Time taken by algorithm ' + str(b - a) + ' seconds')

    # compute the accuracy for training set and testing set
    train_Pred = classify(train_data, parameters)
    print(train_Pred)
    test_Pred = classify(test_data, parameters)

    trAcc = np.mean(np.equal(train_Pred, train_label).astype(int))*100
    teAcc = np.mean(np.equal(test_Pred, test_label).astype(int))*100
    print("Accuracy for training set is {0:0.3f} %".format(trAcc))
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))
    
    #3
    print("ADAM")
    a = time.time()
    costs3, parameters = multi_layer_network(train_data, train_label, test_data, test_label, net_dims, 'adam', num_iterations=num_iterations, learning_rate=learning_rate, batch_size=batch_size)
    plt.plot(iterations, costs3, color='#FFC627')
    b = time.time()
    print('Time taken by algorithm ' + str(b - a) + ' seconds')

    # compute the accuracy for training set and testing set
    train_Pred = classify(train_data, parameters)
    print(train_Pred)
    test_Pred = classify(test_data, parameters)

    trAcc = np.mean(np.equal(train_Pred, train_label).astype(int))*100
    teAcc = np.mean(np.equal(test_Pred, test_label).astype(int))*100
    print("Accuracy for training set is {0:0.3f} %".format(trAcc))
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))
    
    #4
    print("RMS")
    a = time.time()
    costs4, parameters = multi_layer_network(train_data, train_label, test_data, test_label, net_dims, 'rms', num_iterations=num_iterations, learning_rate=learning_rate, batch_size=batch_size)
    plt.plot(iterations, costs4, color='#E9617E')
    b = time.time()
    print('Time taken by algorithm ' + str(b - a) + ' seconds')

    # compute the accuracy for training set and testing set
    train_Pred = classify(train_data, parameters)
    print(train_Pred)
    test_Pred = classify(test_data, parameters)

    trAcc = np.mean(np.equal(train_Pred, train_label).astype(int))*100
    teAcc = np.mean(np.equal(test_Pred, test_label).astype(int))*100
    print("Accuracy for training set is {0:0.3f} %".format(trAcc))
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))
    
    #5
    print("NONE")
    a = time.time()
    costs5, parameters = multi_layer_network(train_data, train_label, test_data, test_label, net_dims, 'none', num_iterations=num_iterations, learning_rate=learning_rate, batch_size=batch_size)
    plt.plot(iterations, costs5, color='#78BE20')
    b = time.time()
    print('Time taken by algorithm ' + str(b - a) + ' seconds')

    
    plt.title("Loss on Fashion MNIST with Initial Learning Rate " +str(learning_rate)+ "and Minibatch-size " +str(batch_size) )
    
    # compute the accuracy for training set and testing set
    train_Pred = classify(train_data, parameters)
    print(train_Pred)
    test_Pred = classify(test_data, parameters)

    trAcc = np.mean(np.equal(train_Pred, train_label).astype(int))*100
    teAcc = np.mean(np.equal(test_Pred, test_label).astype(int))*100
    print("Accuracy for training set is {0:0.3f} %".format(trAcc))
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))
    
    ### Plot costs
    '''iterations = [x*10 for x in range(50)]
    plt.plot(iterations, costs, color='green')'''
    #plt.plot(iterations, costs_val, color='red')
    
    plt.show()

if __name__ == "__main__":
    main()


# In[ ]:




