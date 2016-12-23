from MatrixReading import read_matrix
import numpy as np
from matplotlib import pyplot as plt

def Kernel(x, z, bandwidth=8.0):
    return np.exp(-1/2.0/bandwidth**2*np.linalg.norm(x-z)**2)

def sign(a):
    if a > 0:
        return 1
    else:
        return -1

def predict(training_size):
    data, tokens, categories = read_matrix("spam_data/MATRIX.TRAIN.%s"%training_size)
    data = np.array(data.todense().tolist())
    data = (data > 0)*1
    for i in range(len(categories)):
        if categories[i] == 0:
            categories[i] = -1
    categories = np.array(categories)

    # y = 1 spam
    # y = -1 not spam
    K = np.array([[Kernel(data[i], data[j]) for j in range(data.shape[0])] for i in range(data.shape[0])]) # m x m matrix
    alpha = np.array([[0.0]]*data.shape[0])   # m x 1 matrix
    for epoch in range(40):
        for loop in range(data.shape[0]):
            i = np.random.choice(range(data.shape[0]), 1)[0]
            subgrad = np.array([[0.0]]*K.shape[0])
            if categories[i]*(K[:,i].dot(alpha.reshape((-1)))) < 1:
                subgrad = -categories[i]*K[:,i]
            alpha -= 1.0/np.sqrt(epoch*data.shape[0]+loop+1) * (subgrad.reshape((-1,1)) + 1.0/64*K[:,i].reshape((-1,1))*alpha[i][0])

    test_data, tokens, test_categories = read_matrix("spam_data/MATRIX.TEST")
    test_data = np.array(test_data.todense().tolist())
    test_data = (test_data > 0)*1
    for i in range(len(test_categories)):
        if test_categories[i] == 0:
            test_categories[i] = -1
    test_categories = np.array(test_categories)
    predict_categories = np.array([sign(np.array([Kernel(data[j], test_data[i]) for j in range(data.shape[0])]).dot(alpha)[0]) for i in range(test_data.shape[0])])
    return np.mean(test_categories == predict_categories)


x = [50, 100, 200, 400, 800,1400]
y = [predict(size) for size in x]
print (y)
plt.plot(x, y,'ro')
plt.xlabel("training size")
plt.ylabel("test acc")
plt.show()