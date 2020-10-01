import numpy as np
import matplotlib.pyplot as plt

# Augmented data vectors, (1, x1, x2, cl)
N = 100
cov = np.array([[0.01, 0],
                [0, 0.01]])
m1 = np.array([0., 0.])
m2 = np.array([1., 1.])
m3 = np.array([0., 1.])
m4 = np.array([1., 0.])

p1 = np.hstack((np.ones([N,1]), np.random.multivariate_normal(m1, cov, size=N),
                np.zeros([N,1])))
p2 = np.hstack((np.ones([N,1]), np.random.multivariate_normal(m2, cov, size=N),
                np.zeros([N,1])))
p3 = np.hstack((np.ones([N,1]), np.random.multivariate_normal(m3, cov, size=N),
                np.ones([N,1])))
p4 = np.hstack((np.ones([N,1]), np.random.multivariate_normal(m4, cov, size=N),
                np.ones([N,1])))

data = np.vstack((p1, p2, p3, p4)).T

np.random.shuffle(data.T)



N_test = 50
t1 = np.hstack((np.ones([N_test,1]),
                np.random.multivariate_normal(m1, cov, size=N_test),
                np.zeros([N_test,1])))
t2 = np.hstack((np.ones([N_test,1]),
                np.random.multivariate_normal(m2, cov, size=N_test),
                np.zeros([N_test,1])))
t3 = np.hstack((np.ones([N_test,1]),
                np.random.multivariate_normal(m3, cov, size=N_test),
                np.ones([N_test,1])))
t4 = np.hstack((np.ones([N_test,1]),
                np.random.multivariate_normal(m4, cov, size=N_test),
                np.ones([N_test,1])))
test_data = np.vstack((t1, t2, t3, t4)).T
np.random.shuffle(data.T)




class Multilayer_Perceptron:
    """
    Creates a multilayer perceptron with any given number of neurons and layers
    """
    def __init__(self, X, y, activation_func = None):
        self.X = X
        self.y = y
        self.N = self.X.shape[1]
        if activation_func == None: self.activation_func = self.logistic
        else: self.activation_func = activation_func
        self.weights = []
        self.v_gr = []
        self.y_gr = [self.X.copy()]


    def __reset__(self):
        self.v_gr = []
        self.y_gr = [self.X.copy()]


    def add_layer(self, neuron_N, scale = 5):
        """
        Initiates a random layer to the model
        """
        try:
            l = self.weights[-1].shape[0]
        except IndexError:
            l = self.X.shape[0]

        weights = scale*(np.random.random((neuron_N, l)) - 1/2)
        self.weights.append(weights)


    def test(self, testing_data):
        """
        Tests the classifier on training data
        """
        input = testing_data
        y_ = []
        for j in range(len(self.weights)):
            v = self.weights[j] @ input
            y = self.activation_func(v)
            input = y
            y_.append(y)
        return y_[-1]


    def forward_computation(self):
        """
        Iterates through all layers and calculates v and y.
        weight shape: (p, l)
        input shape:  (l, N)
        output shape: (p, N)
        """
        for j in range(len(self.weights)):
            if len(self.v_gr) == 0:
                input = self.X
            else: input = self.y_gr[-1]

            # Finds the p output values for all N training points.
            v = self.weights[j] @ input
            self.v_gr.append(v)
            self.y_gr.append(self.activation_func(v))


    def backwards_propagation(self, mu = 0.1):
        """
        Computes updated weights by backwards propagation.
        Cost function is defined as the sum of squared errors.
        Minimum is found by gradient descent.
        """
        deltas = []
        new_weights = []
        for r in range(1, len(self.weights)+1):
            # Iteration over layers from L to 1

            weight = np.array(self.weights[-r])
            # Shape (p, N)
            y_neg1 = self.y_gr[-r]
            v_neg1 = self.v_gr[-r]
            # Shape (l, N)
            y_neg2 = self.y_gr[-r-1]

            d_ = []
            for i in range(self.N):
                # Iteration over samples
                # (p, 1) vectors of output values
                y_ = y_neg1[:,i]
                v_ = v_neg1[:,i]
                df = self.activation_func(v_, derivative=True)

                if r == 1:
                    # layer L always with (p, 1) shape, i.e. float objekt
                    e = y_ - self.y[i]

                else:
                    w_ = new_weights[-1]
                    del_ = deltas[-1]
                    # vector for sum in delta over all combinations to k: j.
                    e = np.sum([del_[i,k] * w_[k,:] for k in range(w_.shape[0])], axis=0)

                delta = e*df
                d_.append(delta)

            new_weight = np.zeros(weight.shape)
            # Iterates over neurons j
            for j in range(weight.shape[0]):
                # weighting change
                dw_j = - mu * np.sum([d_[i][j] * y_neg2[:,i] for i in range(self.N)], axis=0)
                new_weight[j] = weight[j] + dw_j
            new_weights.append(new_weight)
            if weight.shape != new_weight.shape: print('Old weight must have same dimensions as new')

            deltas.append(np.matrix(d_))
        self.weights = new_weights[::-1]
        self.__reset__()


    def logistic(self, x, a=1, derivative = False):
        f = 1/(1 + np.exp(-a*x))
        if not derivative:
            return f
        else:
            return np.exp(-a*x) * f**2


    def ReLU(self, x, derivative = False):
        x_ = x.copy()
        if not derivative:
            x_[x < 0] = 0
        else:
            x_[x > 0] = 1
            x_[x < 0] = 0
        return x_


# Design model. Here as a two-layer model with two neurons in the hidden layer
MP = Multilayer_Perceptron(data[:3, :], data[3, :])
MP.add_layer(3)
MP.add_layer(1)
MP.forward_computation()


old = None
epoch = 0
tr_acc = []
te_acc = []
while epoch < 400:
    epoch += 1
    old = MP.weights
    MP.backwards_propagation()
    MP.forward_computation()

    est_y = np.round(MP.y_gr[-1]).astype('int')
    act_y = MP.y.astype('int')
    tr_acc.append(np.sum(est_y == act_y)/est_y.shape[1])

    est_y_te = np.round(MP.test(test_data[:3, :])).astype('int')
    act_y_te = test_data[3, :].astype('int')
    te_acc.append(np.sum(est_y_te == act_y_te)/est_y_te.shape[1])

# Accuracy matrices
L = np.zeros((2,2))
for n in range(data.shape[1]):
    L[act_y[n], est_y[0,n]] += 1

L_te = np.zeros((2,2))
for n in range(test_data.shape[1]):

    L_te[act_y_te[n], est_y_te[0,n]] += 1

print(L)
print(r'Training data accuracy: %s quantile' % str(tr_acc[-1]))

print(L_te)
print(r'Test data accuracy: %s quantile' % str(te_acc[-1]))

# Plot of training- and test accuracy as a function of epochs.
plt.plot(tr_acc, 'r-', label='Accuracy on training data')
plt.plot(te_acc, 'b-', label='Accuracy on testing data')
plt.legend()
plt.ylim(0.4, 1.01)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
