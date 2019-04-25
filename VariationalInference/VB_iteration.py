#  Copyright (c) 2019. University of Oxford

import scipy.special as ss
import numpy as np


def VB_iteration(X, nn_output, alpha_volunteers, alpha0_volunteers):
    """
    performs one iteration of variational inference update for BCCNet (E-step) -- update for approximating posterior of
    true labels and confusion matrices
    N - number of data points
    J - number of true classes
    L - number of classes used by volunteers (normally L == J)
    W - number of volunteers

    :param X: N X W volunteer answers, -1 encodes a missing answer
    :param nn_output: N X J logits (not a softmax output!)
    :param alpha_volunteers: J X L X W - current parameters of posterior Dirichlet for confusion matrices
    :param alpha0_volunteers: J X L -  parameters of the prior Dirichlet for confusion matrix
    :return: q_t - approximating posterior for true labels, alpha_volunteers - updated posterior for confusion matrices,
        lower_bound_likelihood - ELBO
    """

    ElogPi_volunteer = expected_log_Dirichlet_parameters(alpha_volunteers)

    # q_t
    q_t, Njl, rho = expected_true_labels(X, nn_output, ElogPi_volunteer)

    # q_pi_workers
    alpha_volunteers = update_alpha_volunteers(alpha0_volunteers, Njl)

    # Low bound
    lower_bound_likelihood = compute_lower_bound_likelihood(alpha0_volunteers, alpha_volunteers, q_t, rho, nn_output)

    return q_t, alpha_volunteers, lower_bound_likelihood


def logB_from_Dirichlet_parameters(alpha):
    logB = np.sum(ss.gammaln(alpha)) - ss.gammaln(np.sum(alpha))

    return logB


def expected_log_Dirichlet_parameters(param):
    size = param.shape

    result = np.zeros_like(param)

    if len(size) == 1:
        result = ss.psi(param) - ss.psi(np.sum(param))
    elif len(size) == 2:
        result = ss.psi(param) - np.transpose(np.tile(ss.psi(np.sum(param, 1)), (size[1], 1)))
    elif len(size) == 3:
        for i in range(size[2]):
            result[:, :, i] = ss.psi(param[:, :, i]) - \
                              np.transpose(np.tile(ss.psi(np.sum(param[:, :, i], 1)), (size[1], 1)))
    else:
        raise Exception('param can have no more than 3 dimensions')

    return result


def expected_true_labels(X, nn_output, ElogPi_volunteer):
    N, W = X.shape  # N = Number of subjects, W = Number of volunteers.
    J = ElogPi_volunteer.shape[0]  # J = Number of classes
    L = ElogPi_volunteer.shape[1] # L = Number of classes used by volunteers

    rho = np.copy(nn_output)

    for w in range(W):
        inds = np.where(X[:, w] > -1)
        rho[inds, :] = rho[inds, :] + np.transpose(ElogPi_volunteer[:, np.squeeze(X[inds, w]), w])

    rho = rho - np.transpose(np.tile(np.max(rho, 1), (J, 1)))

    q_t = np.exp(rho) / np.maximum(1e-60, np.transpose(np.tile(np.sum(np.exp(rho), 1), (J, 1))))
    q_t = np.maximum(1e-60, q_t)

    Njl = np.zeros((J, L, W), dtype=np.float64)
    for w in range(W):
        for l in range(L):
            inds = np.where(X[:, w] == l)[0]
            Njl[:, l, w] = np.sum(q_t[inds, :], 0)

    return q_t, Njl, rho


def update_alpha_volunteers(alpha0_volunteers, Njl):
    W = alpha0_volunteers.shape[2]
    alpha_volunteers = np.zeros_like(alpha0_volunteers)

    for w in range(W):
        alpha_volunteers[:, :, w] = alpha0_volunteers[:, :, w] + Njl[:, :, w]

    return alpha_volunteers


def compute_lower_bound_likelihood(alpha0_volunteers, alpha_volunteers, q_t, rho, nn_output):
    W = alpha0_volunteers.shape[2]

    ll_pi_worker = 0
    for w in range(W):
        ll_pi_worker = ll_pi_worker - np.sum(logB_from_Dirichlet_parameters(alpha0_volunteers[:, :, w]) -
                                             logB_from_Dirichlet_parameters(alpha_volunteers[:, :, w]))

    ll_t = -np.sum(q_t * rho) + np.sum(np.log(np.sum(np.exp(rho), axis=1)), axis=0)

    ll_nn = np.sum(q_t * nn_output) - np.sum(np.log(np.sum(np.exp(nn_output), axis=1)), axis=0)

    ll = ll_pi_worker + ll_t + ll_nn  # VB lower bound

    return ll


