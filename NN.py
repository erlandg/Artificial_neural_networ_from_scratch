import numpy as np


class Multilayer_Perceptron:
    """
    Creates a multilayer perceptron with any given number of neurons and layers
    """
    def __init__(self, X, y, activation_func = None):
        self.X = X
        self.y = y
        self.N = self.X.shape[1]
        if activation_func == None:
            self.activation_func = self.logistic
        self.weights = []
        self.d_weights = []
        self.v_gr = []
        self.y_gr = [self.X.copy()]
        self.cost = []


    def __reset__(self):
        """
        Resets results after completed epoch.
        """
        self.v_gr = []
        self.y_gr = [self.X.copy()]


    def add_layer(self, neuron_N, scale = 3):
        """
        Initiates a random layer to the model
        """
        try:
            l = self.weights[-1].shape[0]
        except IndexError:
            l = self.X.shape[0]

        weights = scale*(np.random.random((neuron_N, l)) - 1/2)
        self.weights.append(weights)
        # Zero start momentum
        self.d_weights.append(np.zeros(weights.shape))


    def test(self, testing_data, y_if_cost = None):
        """
        Tests the classifier on training data
        returns: output (N_te,), cost = (1,)
        """
        input = testing_data
        y_ = []
        for j in range(len(self.weights)):
            v = self.weights[j] @ input
            input = self.activation_func(v)
            y_.append(input)

        if np.sum(y_if_cost) != None:
            # Calculate cost function of the test data
            cost = 1/2*np.sum((y_[-1] - y_if_cost)**2)
            return y_[-1], cost
        else: return y_[-1]


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

        self.cost.append(1/2*np.sum((self.y_gr[-1] - self.y)**2))


    def backwards_propagation(self, mu = 0.1, momentum=0.7):
        """
        Computes updated weights by backwards propagation.
        Cost function is defined as the sum of squared errors default (SSE).
        May also use cross entropy (CE) as cost function.
        Minimum is found by gradient descent.
        """
        deltas = []
        new_weights = []
        new_d_weights = []
        for r in range(1, len(self.weights)+1):
            # Iteration over layers from L to 1

            weight = np.array(self.weights[-r])
            old_d_w = np.array(self.d_weights[-r])
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

                if r != 1:
                    w_ = new_weights[-1]
                    del_ = deltas[-1]
                    # vector for sum in delta over all combinations to k: j.
                    e = np.sum([del_[i,k] * w_[k,:] for k in range(w_.shape[0])], axis=0)

                else:
                    # layer L always with (p, 1) shape, i.e. float objekt
                    e = y_ - self.y[i]

                delta = e*df
                d_.append(delta)

            new_weight = np.zeros(weight.shape)
            new_d_weight = np.zeros(weight.shape)
            # Iterates over neurons j
            for j in range(weight.shape[0]):
                # weighting change with momentum
                dw_j = momentum * old_d_w[j] - mu * np.sum([d_[i][j] * y_neg2[:,i] for i in range(self.N)], axis=0)
                new_d_weight[j] = dw_j
                new_weight[j] = weight[j] + dw_j
            # Update weights for given layer
            new_weights.append(new_weight)
            new_d_weights.append(new_d_weight)

            deltas.append(np.matrix(d_))
        # Update final weights and final weighting change
        self.weights = new_weights[::-1]
        self.d_weights = new_d_weights[::-1]
        # Ready for next computation
        self.__reset__()


    def logistic(self, x, a=1, derivative = False):
        """
        Our activation function of choice.
        """
        f = 1/(1 + np.exp(-a*x))
        if not derivative:
            return f
        else:
            return np.exp(-a*x) * f**2
