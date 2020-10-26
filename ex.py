import matplotlib.pyplot as plt
from NN import *

# Simulation of data
# Augmented data vectors, (1, x1, x2, cl)
N = 100
var = 0.1
cov = np.array([[var, 0],
                [0, var]])
m1 = np.array([0., 0.])
m2 = np.array([1., 1.])
m3 = np.array([0., 1.])
m4 = np.array([1., 0.])

p1 = np.hstack((np.ones([N,1]),
                np.random.multivariate_normal(m1, cov, size=N),
                np.zeros([N,1])))
p2 = np.hstack((np.ones([N,1]),
                np.random.multivariate_normal(m2, cov, size=N),
                np.zeros([N,1])))
p3 = np.hstack((np.ones([N,1]),
                np.random.multivariate_normal(m3, cov, size=N),
                np.ones([N,1])))
p4 = np.hstack((np.ones([N,1]),
                np.random.multivariate_normal(m4, cov, size=N),
                np.ones([N,1])))

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(p1[:,1], p1[:,2], 'r.')
ax1.plot(p2[:,1], p2[:,2], 'r.')
ax1.plot(p3[:,1], p3[:,2], 'b.')
ax1.plot(p4[:,1], p4[:,2], 'b.')

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



# Classification of data
# Design model. Here as a two-layer model with two neurons in the hidden layer
MP = Multilayer_Perceptron(data[:3, :], data[3, :])
MP.add_layer(3)
MP.add_layer(1)
MP.forward_computation()


old = None
max_epoch = 200
epoch = 0
tr_acc = []
te_acc = []
# Progress of cost function
te_cost = []
while epoch < max_epoch:
    epoch += 1
    old = MP.weights
    # Update weights
    MP.backwards_propagation()
    # New results
    MP.forward_computation()

    est_y = np.round(MP.y_gr[-1]).astype('int')
    act_y = MP.y.astype('int')
    tr_acc.append(np.sum(est_y == act_y)/est_y.shape[1])

    est_y_te, test_cost = MP.test(test_data[:3, :], test_data[3, :])
    est_y_te = np.round(est_y_te).astype('int')
    act_y_te = test_data[3, :].astype('int')
    te_acc.append(np.sum(est_y_te == act_y_te)/est_y_te.shape[1])
    te_cost.append(test_cost)


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
ax22 = ax2.twinx()
ax2.plot(tr_acc, 'r-', label='Accuracy on training data')
ax2.plot(te_acc, 'b-', label='Accuracy on testing data')
ax2.set_ylim(0.4, 1.01)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')

ax22.plot(MP.cost, 'g-', label='Training data cost function')
ax22.plot(te_cost, 'purple', label='Testing data cost')
ax22.set_ylabel('Cost function')

fig.subplots_adjust(bottom=0.2)
fig.legend(loc='lower right')
plt.show()
