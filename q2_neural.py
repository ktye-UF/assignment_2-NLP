import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    input_layer = data
    N = data.shape[0]
    
    hidden_layer = np.dot(data,W1) + np.tile(b1,(N,1))   # input layer to hidden layer
    hidden_layer_value = sigmoid(hidden_layer)           # output of hidden layer after applying sigmoid function
    
    output_layer = np.dot(hidden_layer_value,W2) + np.tile(b2,(N,1)) # from hidden to output
    output_layer_value = softmax(output_layer)           # output of output layer after applying softmax function
    
    # cost function
    cost = - np.sum(labels * np.log(output_layer_value)) # cost function when using the cross entropy cost & softmax

    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation
    
    output_delta = (output_layer_value - labels)         # sensitivity when using the cross entropy cost
    
    gradW2 = np.dot(hidden_layer_value.T,output_delta)   # calculate W2 gradient
    gradb2 = np.sum(output_delta,axis=0)                 # calculate b2 gradient

    input_delta = np.dot(output_delta,W2.T) * sigmoid_grad(hidden_layer_value) # gradient back to input layer
    
    gradW1 = np.dot(input_layer.T,input_delta)           # calculate W1 gradient
    gradb1 = np.sum(input_delta,axis=0)                  # calculate b1 gradient

    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
