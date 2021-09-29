import numpy as np
import matplotlib.pyplot as plt
def computeCost(X,y,theta):
    """
    Take in a numpy array X,y, theta and generate the cost function using theta as parameter in a linear regression model
    """
    # np.dot(A,B) => gives dot product of A and B
    # np.power(A,n) => returns array with each element raised to the power n
    # np.sum(A) => Returns scalar with every element in A summed up
    m=len(y)
    # print(X)
    # print(theta)
    predictions= np.dot(X,theta)
    # z=X[:,1]
    # plt.scatter(z, predictions) 
    # plt.show()
    # print(predictions)
    # print(y)
    square_err= np.power(np.subtract(predictions,y),2)
    return 1/(2*m) * np.sum(square_err)