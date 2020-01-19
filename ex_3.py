import numpy as np
import matplotlib.pyplot as plt
import random

# hyper parameters
HIDDEN = 80
CLASSES = 10
ETA = 0.001
EPOCHS = 20
INPUT = 28 * 28


def softmax(x):
    c = np.max(x)
    ex = np.exp(x - c)
    x1 = ex / np.sum(ex)
    return x1


def ReLU(x):
    return np.maximum(0, x)


def ReLU_dx(x):
    return (x > 0).astype(np.float)

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for x, label in dataset:
        y_hat = predict(x, params)
        if y_hat == label:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def predict(x, params):
    W, b, U, b2 = [params[key] for key in ('W', 'b', 'U', 'b2')]
    z1 = np.dot(W, x) + b
    h1 = ReLU(z1)
    z2 = np.dot(U, h1) + b2
    h2 = softmax(z2)
    return np.argmax(h2)

#forword propagation
def classifier_output(x, y, params):
    W, b, U, b2 = [params[key] for key in ('W', 'b', 'U', 'b2')]
    z1 = np.dot(W, x) + b
    h1 = ReLU(z1)
    z2 = np.dot(U, h1) + b2
    h2 = softmax(z2)
    new_params = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2}
    for key in params:
        new_params[key] = params[key]
    return new_params, h2


def calc_gradiants(grads):
    x, y, z1, h1, z2, h2 = [grads[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2')]
    dU = np.outer(h2, h1)
    dU[y] -= h1
    db2 = np.copy(h2)
    db2[y] -= 1

    dz2 = h2.dot(grads['U']) - grads['U'][y, :]
    dh1 = ReLU_dx(z1)
    db = dz2 * dh1
    dW = np.outer(db, x)
    return {'b': db, 'W': dW, 'b2': db2, 'U': dU}


def train_classifier(train_data,params):
    # use for initialize the net
    #train_accuracy_data = np.zeros(EPOCHS)
    #dev_accuray_data = np.zeros(EPOCHS)

    for I in range(EPOCHS):
        cum_loss = 0.0
        random.shuffle(train_data)
        for x, y in train_data:
            params_update, prediction = classifier_output(x, y, params)
            loss = -np.log(prediction[y])
            cum_loss += loss
            gradients = calc_gradiants(params_update)
            for key in params:
                params[key] -= ETA * gradients[key]

        # calculate loss and accuracy on the validation
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        #dev_accuracy = accuracy_on_dataset(dev_data, params)
        #train_accuracy_data[I] = train_accuracy
        #dev_accuray_data[I] = dev_accuracy
        print(I, train_loss, train_accuracy)

    # use for initialize
    #plt.plot(range(EPOCHS), train_accuracy_data)
    #plt.plot(range(EPOCHS), dev_accuray_data)
    #plt.show()
    return params

#Create classifer parameters with initialize of uniform distribution
def create_classifier(in_dim, hid_dim, out_dim):
    lower_bound = -0.05
    upper_bound = 0.05
    W = np.random.uniform(lower_bound, upper_bound, (hid_dim, in_dim))
    b = np.random.uniform(lower_bound, upper_bound, (hid_dim,))
    U = np.random.uniform(lower_bound, upper_bound, (out_dim, hid_dim))
    b2 = np.random.uniform(lower_bound, upper_bound,(out_dim,))

    return {'W': W, 'b': b, 'U': U, 'b2': b2}

#predict the labels of the test set
def classify_test(test_x, params):
    with open("test_y", "w") as test_labels:
        # go over all the test
        for xt in test_x:
            y_hat = predict(xt, params)
            test_labels.write(str(y_hat))
            test_labels.write("\n")


if __name__ == '__main__':
    import sys

    train_x = np.loadtxt(sys.argv[1], dtype='int') / 255
    train_y = np.loadtxt(sys.argv[2], dtype='int')
    test_x = np.loadtxt(sys.argv[3]) / 255

    params = create_classifier(INPUT, HIDDEN, CLASSES)
    train_data = list(zip(train_x, train_y))

    params = train_classifier(train_data,params)
    classify_test(test_x, params)

#For self use
'''
    from sklearn.model_selection import train_test_split

    train_x, validation_x, train_y, validation_y = train_test_split(
        train_x, train_y, test_size=0.33, random_state=42)
            #dev_data = list(zip(validation_x, validation_y))
'''