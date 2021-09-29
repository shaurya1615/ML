import numpy as np
from sigmoid import sigmoid


def costFunction(theta, X, y):
    '''returns cost for theta, X and y
    np.log(a)==> returns array with elementwise log on array a
    use the sigmoid function that's being imported above 
    '''
    m = y.size
    h = sigmoid(np.dot(X,theta))

    J = -1*1/m*(np.dot(np.log(h).T,y)+np.dot(np.log(1-h),(1-y)))           #cost function
    # print(theta.reshape(-1,1).shape)

    # print(theta.shape)
    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])


def gradient(theta, X, y):
	'''' calculate gradient descent for logistic regression'''
	m = y.size
    
	h = sigmoid(X.dot(theta.reshape(-1,1)))     #theta is a vector so converting it to a matric of (3, 1)
    
	grad = (1/m)*(X.T.dot(h-y))  

	return(grad.flatten())			# returns copy of array in one dimension
