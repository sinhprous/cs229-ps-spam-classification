from MatrixReading import read_matrix
import numpy as np
from matplotlib import pyplot as plt

def predict(training_size):
    data, tokens, categories = read_matrix("spam_data/MATRIX.TRAIN.%s"%training_size)
    data = np.array(data.todense().tolist())
    categories = np.array(categories)

    # y = 1 spam
    # y = 0 not spam

    num_spam_doc = np.sum(categories == [1]*len(categories))
    num_normal_doc = np.sum(categories == [0]*len(categories))
    num_spam_tok = np.sum([np.sum(data[i]) for i in range(data.shape[0]) if categories[i]==1])
    num_normal_tok = np.sum([np.sum(data[i]) for i in range(data.shape[0]) if categories[i]==0])

    phi_spam = [(np.sum([data[i][k] for i in range(data.shape[0]) if categories[i]==1])+1)/(num_spam_tok+data.shape[1]) for k in range(data.shape[1])]
    phi_normal = [(np.sum([data[i][k] for i in range(data.shape[0]) if categories[i]==0])+1)/(num_normal_tok+data.shape[1]) for k in range(data.shape[1])]
    phi_y = float(num_spam_doc)/len(categories)

    test_data, tokens, test_categories = read_matrix("spam_data/MATRIX.TEST")
    test_data = np.array(test_data.todense().tolist())
    tokens = np.array(tokens)
    test_categories = np.array(test_categories)

    predict_spam_posterior = [np.sum(test_data[i]*np.log(phi_spam))+np.log(phi_y) for i in range(test_data.shape[0])]
    predict_normal_posterior = [np.sum(test_data[i]*np.log(phi_normal))+np.log(1-phi_y) for i in range(test_data.shape[0])]
    predict_categories = np.array(predict_spam_posterior) >= np.array(predict_normal_posterior)
    return np.mean(predict_categories == test_categories)

#print ("test accuracy: %f"%np.mean(predict_categories == test_categories))

#indi = np.log(np.array(phi_spam)/np.array(phi_normal))
#print (tokens[np.argsort(indi)[-5:]])

x = [50, 100, 200, 400, 800,1400]
y = [predict(size) for size in x]
plt.plot(x, y,'ro')
plt.xlabel("training size")
plt.ylabel("test acc")
plt.show()