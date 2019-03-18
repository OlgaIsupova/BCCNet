import scipy.special as ss
import numpy as np


def VB_iteration(X, nn_output, alpha_workers, alpha0_workers):
    """
    performs one iteration of variational inference update for BCCNet (E-step) -- update for approximating posterior of
    true labels and confusion matrices
    N - number of data points
    J - number of true classes
    L - number of classes used by experts (normally L == J)
    W - number of experts

    :param X: N X W expert answers, -1 encodes a missing answer
    :param nn_output: N X J logits (not a softmax output)
    :param alpha_workers: J X L X W - current parameters of posterior Dirichlet for confusion matrices
    :param alpha0_workers: J X L -  parameters of the prior Dirichlet for confusion matrix
    :return: q_t - approximating posterior for true labels, alpha_workers - updated posterior for confusion matrices,
        lower_bound_likelihood - ELBO
    """

    ElogPi_worker = expected_log_Dirichlet_parameters(alpha_workers)

    # q_t
    q_t, Njl, rho = expected_true_labels(X, nn_output, ElogPi_worker)

    # q_pi_workers
    alpha_workers = updated_alpha_workers(alpha0_workers, Njl)

    # Low bound
    lower_bound_likelihood = compute_lower_bound_likelihood(alpha0_workers, alpha_workers, q_t, rho, nn_output)

    return q_t, alpha_workers, lower_bound_likelihood


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


def expected_true_labels(X, nn_output, ElogPi_worker):
    N, W = X.shape  # N = Number of subjects, W = Number of volunteers.
    J = ElogPi_worker.shape[0]  # J = Number of classes
    L = ElogPi_worker.shape[1] # L = Number of classes used by experts

    rho = np.copy(nn_output)

    for w in range(W):
        inds = np.where(X[:, w] > -1)
        rho[inds, :] = rho[inds, :] + np.transpose(ElogPi_worker[:, np.squeeze(X[inds, w]), w])

    rho = rho - np.transpose(np.tile(np.max(rho, 1), (J, 1)))

    q_t = np.exp(rho) / np.maximum(1e-60, np.transpose(np.tile(np.sum(np.exp(rho), 1), (J, 1))))
    q_t = np.maximum(1e-60, q_t)

    Njl = np.zeros((J, L, W), dtype=np.float64)
    for w in range(W):
        for l in range(L):
            inds = np.where(X[:, w] == l)[0]
            Njl[:, l, w] = np.sum(q_t[inds, :], 0)

    return q_t, Njl, rho


def updated_alpha_workers(alpha0_workers, Njl):
    W = alpha0_workers.shape[2]
    alpha_workers = np.zeros_like(alpha0_workers)

    for w in range(W):
        alpha_workers[:, :, w] = alpha0_workers[:, :, w] + Njl[:, :, w]

    return alpha_workers


def compute_lower_bound_likelihood(alpha0_workers, alpha_workers, q_t, rho, nn_output):
    W = alpha0_workers.shape[2]

    ll_pi_worker = 0
    for w in range(W):
        ll_pi_worker = ll_pi_worker - np.sum(logB_from_Dirichlet_parameters(alpha0_workers[:, :, w]) -
                                             logB_from_Dirichlet_parameters(alpha_workers[:, :, w]))

    ll_t = -np.sum(q_t * rho) + np.sum(np.log(np.sum(np.exp(rho), axis=1)), axis=0)

    ll_nn = np.sum(q_t * nn_output) - np.sum(np.log(np.sum(np.exp(nn_output), axis=1)), axis=0)

    ll = ll_pi_worker + ll_t + ll_nn  # VB lower bound

    return ll


