import numpy as np
import copy
import matplotlib.pyplot as plt

def initialize_parameters_deep(layer_dims):
    """"
    layer_dims = [n_x, n_h, ... ,n_y]
    ex : [12288, 20, 20, ... , 20, 1]
    """
    params = {}
    L = len(layer_dims)
    for l in range(1, L):
        params["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        params["b" + str(l)] = np.zeros((layer_dims[l],1))
    return params
def relu(z):
    return np.maximum(z,0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_with_activation(a_prev, w, b, activation = "relu"):
    z = w@a_prev + b

    activation_cache = z
    linear_cache = (a_prev, w, b)
    cache = (linear_cache, activation_cache)

    if activation == "relu":
        A = relu(z)
    elif activation == "sigmoid":
        A = sigmoid(z)
    return A, cache

    
    
def forward(X, params):
    caches = []
    A = X
    L = len(params) //2

    for l in range(1, L):
        a_prev = A
        A, cache = forward_with_activation(a_prev, params["W" + str(l)], params["b" + str(l)], activation = "relu")
        caches.append(cache)
    AL , cache = forward_with_activation(A, params["W" + str(L)], params["b" + str(L)], activation = "sigmoid")
    caches.append(cache)
    return AL, caches 
    
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = np.sum(Y * np.log(AL) + (1 -Y)* np.log(1-AL)) / -m
    cost = np.squeeze(cost)
    return cost

def relu_backward(dA, activation_cache):
    '''
    activation_cache = z
    '''
    z = activation_cache
    dz = np.array(dA, copy=True)

    dz[z<0] = 0
    return dz

def sigmoid_backward(dA, activation_cache):
    '''
    activation_cahce = z
    '''
    z = activation_cache
    s = 1 / (1+ np.exp(-z))
    dz = dA * s * (1-s)
    return dz 


def backward(dz, linear_cache):
    a_prev, W, b = linear_cache
    m = a_prev.shape[1]
    dW = (dz @ a_prev.T) / m
    db = np.sum(dz, axis = 1, keepdims = True)/ m
    dA_prev = W.T @ dz
    return dA_prev, dW, db


def backward_with_activation(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dz = relu_backward(dA,activation_cache) # activation_cache = z
        dA_prev, dW, db = backward(dz,linear_cache)
    elif activation == "sigmoid":
        dz = sigmoid_backward(dA, activation_cache) # activation_cache = z
        dA_prev, dW, db = backward(dz, linear_cache)
    return dA_prev, dW, db
    
    
def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    #AL부터 시작하는 backward
    dAL = - (np.divide(Y,AL) - np.divide((1-Y),(1-AL)))

    current_cache = caches[-1]
    dA_prev_temp, dW_temp, db_temp = backward_with_activation(dAL, current_cache, activation="sigmoid")

    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = backward_with_activation(dA_prev_temp, current_cache, activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp

    return grads
    
def update_params(params, grads, lr= 0.03):
    parameters = copy.deepcopy(params)

    L = len(params) // 2

    # w = w - lr * wb
    for l in range(1, L+1):
        parameters["W"+str(l)] = parameters["W"+str(l)] - lr * grads["dW" + str(l)]
        parameters["b"+str(l)] = parameters["b"+str(l)] - lr * grads["db" + str(l)]
    return parameters

    
def plot_costs(costs, lr):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(lr))
    plt.show()
    pass
