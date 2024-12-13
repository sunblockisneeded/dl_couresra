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

def forward_with_activation(a_prev, w, b, activation = "relu", keep_prob= 1):
    z = w@a_prev + b

    if activation == "relu":
        A = relu(z)
    elif activation == "sigmoid":
        A = sigmoid(z)

    if keep_prob < 1:
        dropout_vector = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
        A = A * dropout_vector / keep_prob

    activation_cache = z
    if keep_prob == 1:
        linear_cache = (a_prev, w, b)
    elif keep_prob < 1:
        linear_cache = (a_prev, w, b, dropout_vector)
    cache = (linear_cache, activation_cache)
    return A, cache

    
def forward(X, params, keep_prob =1):
    caches = []
    A = X
    L = len(params) //2

    for l in range(1, L):
        a_prev = A
        A, cache = forward_with_activation(a_prev, params["W" + str(l)], params["b" + str(l)], activation = "relu", keep_prob = keep_prob)
        caches.append(cache)
    AL , cache = forward_with_activation(A, params["W" + str(L)], params["b" + str(L)], activation = "sigmoid", keep_prob =1)
    caches.append(cache)
    return AL, caches 
    
def compute_cost(AL, Y, l2_reg = False, params = None, lambd_ = 10e-4):
    '''
    if l2_reg == True , you should enter params = parameters
    '''

    m = Y.shape[1]
    cost = np.sum(Y * np.log(AL) + (1 -Y)* np.log(1-AL)) / -m


    if l2_reg and params is not None:
        L = len(params) //2
        sum_W = 0
        for l in range(1,L+1):
            tempW = params["W" + str(l)]
            sum_W += np.sum(tempW **2)
        cost += lambd_/(2*m) * sum_W
    elif l2_reg == True and params is None:
        raise ValueError("enter params = parameters")
    cost = np.squeeze(cost)
    return cost

def relu_backward(dA, cache, keep_prob = 1):
    '''
    activation_cache = z
    '''
    linear_cache, activation_cache = cache
    z = activation_cache
    if keep_prob != 1:
        _1,_2,_3,dropout_vector = linear_cache
        dz = np.array(dA, copy= True) * dropout_vector
        dz = dz / keep_prob
    else:
        dz = np.array(dA, copy=True)

    dz[z<0] = 0
    return dz

def sigmoid_backward(dA, cache, keep_prob = 1):
    '''
    activation_cahce = z
    '''
    linear_cache, activation_cache = cache
    z = activation_cache
    #do not dropout at sigmoid
    s = 1 / (1+ np.exp(-z))
    dz = dA * s * (1-s)
    return dz 


def backward(dz, linear_cache, l2_reg = False, lambd_ = 10e-4, keep_prob = 1):
    if keep_prob == 1:
        a_prev, W, b,= linear_cache
    else:
        a_prev, W, b,dropout_vector = linear_cache
    m = a_prev.shape[1]
    dW = (dz @ a_prev.T) / m
    if l2_reg:
        dW += lambd_ / m * W
    db = np.sum(dz, axis = 1, keepdims = True)/ m
    dA_prev = W.T @ dz
    return dA_prev, dW, db


def backward_with_activation(dA, cache, activation, l2_reg = False, lambd_ = 10e-4, keep_prob = 1):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dz = relu_backward(dA,cache, keep_prob) # activation_cache = z
        dA_prev, dW, db = backward(dz,linear_cache,l2_reg, lambd_, keep_prob = keep_prob)
    elif activation == "sigmoid":
        dz = sigmoid_backward(dA, cache,) # activation_cache = z
        dA_prev, dW, db = backward(dz, linear_cache,l2_reg, lambd_, keep_prob = keep_prob)
    return dA_prev, dW, db
    
    
def L_model_backward(AL, Y, caches, l2_reg = False, lambd_ = 10e-4,keep_prob = 1):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    #AL부터 시작하는 backward
    dAL = - (np.divide(Y,AL) - np.divide((1-Y),(1-AL)))

    current_cache = caches[-1]
    dA_prev_temp, dW_temp, db_temp = backward_with_activation(dAL, current_cache, activation="sigmoid" , l2_reg= l2_reg, lambd_ = lambd_) # do not apply keep_prob

    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L-1)):
        current_cache = caches[l]                                  #↓dAL로 되어있는거 조심!
        dA_prev_temp, dW_temp, db_temp = backward_with_activation(dA_prev_temp, current_cache, activation="relu", l2_reg= l2_reg, lambd_= lambd_, keep_prob = keep_prob)
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

def initialize_parameters_type(layer_dims, init_type="he"):
    parameters = {}
    L = len(layer_dims)
    if init_type == "zeros":
        for l in range(1,L):
            parameters["W" + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
            parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    elif init_type =="random":
        for l in range(1,L):
             parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
             parameters["b" + str(l)] = np.zeros((layer_dims[l],1))
    elif init_type =="he":
        for l in range(1,L):
             parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
             parameters["b" + str(l)] = np.zeros((layer_dims[l],1))
    else:
        raise ValueError("select correct parameters init_type. 'zeros'/'random'/'he'")    
    return parameters

def predict(test_x, params):
    AL, _ = forward(test_x,params,keep_prob = 1)
    del _
    y_hat = (AL>=0.5).astype(int)
    return y_hat