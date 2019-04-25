#  Copyright (c) 2019. University of Oxford

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

from NNArchitecture.lenet5_mnist import cnn_for_mnist
from SyntheticCrowdsourcing.synthetic_crowd_volunteers import generate_volunteer_labels
from VariationalInference.VB_iteration import VB_iteration
from utils.utils_dataset_processing import shrink_arrays
from VariationalInference import confusion_matrix

rseed = 1000
np.random.seed(rseed)
tf.set_random_seed(rseed)

# parameters
n_classes = 10
crowdsourced_labelled_train_data_ratio = 0.5
n_crowd_members = 4
crowd_member_reliability_level = 0.6
confusion_matrix_diagonal_prior = 1e-1
n_epoch = 100
batch_size = 32
convergence_threshold = 1e-6

# load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path=os.getcwd() + '/mnist.npz')

# expand images for a cnn
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

# select subsample of train data to be "labelled" by crowd members
labelled_train, whole_train = shrink_arrays([x_train, y_train], crowdsourced_labelled_train_data_ratio, is_shuffle=True)
x_labelled_train = labelled_train[0]
y_labelled_train = labelled_train[1]
x_train = whole_train[0]
y_train = whole_train[1]

# generate synthetic crowdsourced labels
crowdsourced_labels = generate_volunteer_labels(n_volunteers=n_crowd_members, n_classes=n_classes, gt_labels=y_labelled_train,
                                                n_total_tasks=x_train.shape[0],
                                                reliability_level=crowd_member_reliability_level)

# set up a neural net
cnn_model = cnn_for_mnist()

# set up variational parameters
prior_param_confusion_matrices = confusion_matrix.initialise_prior(n_classes=n_classes, n_volunteers=n_crowd_members,
                                                                   alpha_diag_prior=confusion_matrix_diagonal_prior)
variational_param_confusion_matrices = np.copy(prior_param_confusion_matrices)

# initial variational inference iteration (initialisation of approximating posterior of true labels)
initial_nn_output_for_vb_update = np.random.randn(x_train.shape[0], n_classes)

q_t, variational_param_confusion_matrices, lower_bound = \
    VB_iteration(crowdsourced_labels, initial_nn_output_for_vb_update, variational_param_confusion_matrices,
                 prior_param_confusion_matrices)

old_lower_bound = lower_bound

# set up evaluation arrays
nn_training_accuracy = np.zeros((n_epoch,), dtype=np.float64)   # based on predictions from neural network
posterior_estimate_training_accuracy = np.zeros((n_epoch,), dtype=np.float64)   # based on approximated posterior for true labels
nn_test_accuracy = np.zeros((n_epoch,), dtype=np.float64)

# main cycle of training
for epoch in range(n_epoch):
    print(f'epoch {epoch}:')

    # update of parameters of the neural network
    cnn_model.fit(x_train, q_t, epochs=1, shuffle=True, batch_size=batch_size, verbose=0)

    # update of approximating posterior for the true labels and confusion matrices
    # get current predictions from a neural network
    nn_output_for_vb_update = cnn_model.predict(x_train)
    # for numerical stability
    nn_output_for_vb_update = nn_output_for_vb_update - \
        np.tile(np.expand_dims(np.max(nn_output_for_vb_update, axis=1), axis=1), (1, nn_output_for_vb_update.shape[1]))

    q_t, variational_param_confusion_matrices, lower_bound = \
        VB_iteration(crowdsourced_labels, nn_output_for_vb_update, variational_param_confusion_matrices,
                     prior_param_confusion_matrices)

    # evaluation
    nn_training_accuracy[epoch] = np.mean(np.argmax(nn_output_for_vb_update, axis=1) == y_train)
    print(f'\t nn training accuracy: {nn_training_accuracy[epoch]}')

    posterior_estimate_training_accuracy[epoch] = np.mean(np.argmax(q_t, axis=1) == y_train)
    print(f'\t posterior estimate training accuracy: {posterior_estimate_training_accuracy[epoch]}')

    nn_test_prediction = cnn_model.predict(x_test)
    nn_test_accuracy[epoch] = np.mean(np.argmax(nn_test_prediction, axis=1) == y_test)
    print(f'\t nn test accuracy: {nn_test_accuracy[epoch]}')

    # check convergence
    if np.abs((lower_bound - old_lower_bound) / old_lower_bound) < convergence_threshold:
        break

    old_lower_bound = lower_bound

# save weights
cnn_model.save_weights(os.getcwd() + '/trained_weights')

# plotting
plt.plot(range(epoch), nn_training_accuracy[:epoch], label='nn train')
plt.plot(range(epoch), posterior_estimate_training_accuracy[:epoch], label='posterior train')
plt.plot(range(epoch), nn_test_accuracy[:epoch], label='nn test')
plt.legend()
plt.show()







