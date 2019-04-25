#  Copyright (c) 2019. University of Oxford

import numpy as np


def initialise_prior(n_classes, n_volunteers, alpha_diag_prior):
    """
    Create confusion matrix prior for every volunteer - the same prior for each volunteer
    :param n_classes: number of classes (int)
    :param n_volunteers: number of crowd members (int)
    :param alpha_diag_prior: prior for confusion matrices is assuming reasonable crowd members with weak dominance of a
    diagonal elements of confusion matrices, i.e. prior for a confusion matrix is a matrix of all ones where
    alpha_diag_prior is added to diagonal elements (float)
    :return: numpy nd-array of the size (n_classes, n_classes, n_volunteers)
    """
    alpha_volunteer_template = np.ones((n_classes, n_classes), dtype=np.float64) + alpha_diag_prior * np.eye(n_classes)
    return np.tile(np.expand_dims(alpha_volunteer_template, axis=2), (1, 1, n_volunteers))
