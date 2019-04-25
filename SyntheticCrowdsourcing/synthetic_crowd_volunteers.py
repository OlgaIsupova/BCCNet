#  Copyright (c) 2019, University of Oxford

import numpy as np


def generate_confusion_matrices_for_crowd_volunteers_from_Dirichlet(n_volunteers, n_classes, reliability_level=0.6):
    """
    Generates confusion matrices for each of n_volunteers volunteers and n_classes class labels.
    reliability_level determines the expected probability of a volunteer being correct:
    the more reliability_level the less a volunteer is expected to make mistakes, whereas
    reliability_level = 1 will produce always correct volunteers.
    All volunteer confusion matrices generated as samples from the same Dirichlet priors with parameters equal
    to matrices with reliability_level on the diagonal and (1 - reliability_level) spread across uniformly onto
    corresponding the non-diagonal elements of the same row.
    :param n_volunteers: number of crowd members (int)
    :param n_classes: number of class labels (int)
    :param reliability_level: diagonal elements of the prior Dirichlet distribution for confusion matrices (float)
    :return: numpy nd-array of the size (n_classes, n_classes, n_volunteers) of generated confusion matrices
    """

    # Dirichlet prior
    alpha = (reliability_level - (1 - reliability_level) / (n_classes - 1)) * np.eye(n_classes) + \
            (1 - reliability_level) / (n_classes - 1)

    confusion_matrices = np.zeros((n_classes, n_classes, n_volunteers), dtype=np.float64)
    for volunteer in np.arange(n_volunteers):
        for class_label in np.arange(n_classes):
            confusion_matrices[class_label, :, volunteer] = np.random.dirichlet(alpha[class_label, :])

    return confusion_matrices


def generate_confusion_matrices_for_crowd_volunteers(n_volunteers, n_classes, reliability_level=0.6):
    """
    Generates confusion matrices for each of n_volunteers volunteers and n_classes class labels.
    reliability_level determines the expected probability of a volunteer being correct:
    the more reliability_level the less a volunteer is expected to make mistakes. All volunteer confusion matrices
    generated randomly such that on average elements are equal to reliability_level on the diagonal and
    (1 - reliability_level) spread across uniformly onto the corresponding non-diagonal elements of the same row.
    :param n_volunteers: number of crowd members (int)
    :param n_classes: number of class labels (int)
    :param reliability_level: on average diagonal elements of the confusion matrices (float)
    :return: numpy nd-array of the size (n_classes, n_classes, n_volunteers) of generated confusion matrices
    """
    diag_element = (0.5 * n_classes * reliability_level - 0.5) / (1 - reliability_level)
    confusion_matrices = np.zeros((n_classes, n_classes, n_volunteers), dtype=np.float64)
    for volunteer in range(n_volunteers):
        confusion_matrices[:, :, volunteer] = np.random.rand(n_classes, n_classes) + diag_element * np.eye(n_classes)
        row_sums = confusion_matrices[:, :, volunteer].sum(axis=1)
        confusion_matrices[:, :, volunteer] = confusion_matrices[:, :, volunteer] / row_sums[:, np.newaxis]

    return confusion_matrices


def generate_volunteer_answers_from_cm(confusion_matrices, gt_labels, p_fill_task=0.8):
    """
    Generates crowdsourced labels based on input confusion_matrices for each crowd member, ground truth labels gt_labels
    and a probability to fill a task p_fill_task. Ground truth labels gt_labels determines rows of the confusion
    matrices for each crowd member and their answers are sampled according to these probability of discrete
    distribution. Those answers are further vanished with probability p_fill_task that a volunteer has not provided
    an answer for this task.
    :param confusion_matrices: np.array(n_classes, n_classes, n_volunteers) is numpy nd-array with dimensions:
    number of class labels n_classes, number of class labels n_classes, number of crowd members n_volunteers, where
    confusion_matrices[:, :, i] is a confusion matrix for volunteer i
    :param gt_labels: np.array(number of tasks) is numpy array with the length number of tasks, where each element is a
    correct class label for a task, i.e. integer in range(number of classes)
    :param p_fill_task: a probability of each volunteer to perform a task, i.e. a float number in [0, 1]
    :return: labels - np.array(n_tasks, n_volunteers) is numpy nd-array with dimensions: number of tasks n_tasks,
    number of crowd members n_volunteers, where labels[i, j] is an answer for task i by volunteer j. If there is no
    answer, labels[i, j] = -1
    """

    n_tasks = gt_labels.shape[0]
    n_volunteers = confusion_matrices.shape[2]

    labels = np.zeros((n_tasks, n_volunteers), dtype=np.int)

    # sample volunteer answers
    for volunteer in np.arange(n_volunteers):
        labels[:, volunteer] = np.sum(
            np.cumsum(confusion_matrices[gt_labels, :, volunteer], axis=1) < np.random.rand(n_tasks, 1), axis=1)

    # vanish unfilled tasks
    unfilled_mask = np.random.binomial(1, p_fill_task, (n_tasks, n_volunteers)) == 0
    labels[unfilled_mask] = -1

    return labels


def expand_volunteer_labels(labels, final_number_samples):
    """
    Expands a matrix of crowdsourced labels such that the total number of data points is equal to
    final_number_samples filling values with -1 (missing values). Used to simulate a situation when crowd members label
    only a part of data.
    In contrast to p_fill_task parameter in the function generate_volunteer_answers_from_cm which determines a
    probability of each crowd member to label a data point independently, here all additional data points
    (final_number_samples - labels.shape[0]) are not labelled by any of crowd members
    :param labels: current crowdsourced label matrix, numpy nd-array of the size
    (number of data points that could have been labelled by a crowd member, number of crowd members)
    :param final_number_samples: total number of data points among which only labels.shape[0] could have been labelled
    by crowd members (int)
    :return: numpy nd-array of expanded crowdsourced label matrix filled with -1 of the size
    (final_number_samples, labels.shape[1])
    """
    exp_labels = np.pad(labels, ((0, final_number_samples - labels.shape[0]), (0, 0)),
                        mode='constant', constant_values=-1)

    return exp_labels


def generate_volunteer_labels(n_volunteers, n_classes, gt_labels, n_total_tasks=None, reliability_level=0.6,
                              p_fill_task=0.8):
    """
    Performs a full generation of crowdsourced labels procedure
    :param n_volunteers: number of crowd members (int)
    :param n_classes: number of class labels (int)
    :param gt_labels: is numpy array with the length equal to a number of data points that could have been labelled by a
    crowd member, where each element is a correct class label for a data point, i.e. an integer in
    range(number of classes)
    If there are data points which should not be labelled by any of a crowd member, gt_labels should be provided only
    for data points that could be labelled by crowd members.
    :param n_total_tasks: total number of data points. If n_total_tasks > gt_labels.shape[0], the output crowdsourced
    label matrix is filled with missing labels from all crowd members for data points with indices
    gt_labels.shape[0], ..., n_total_tasks - 1. If n_total_tasks=None, n_total_tasks is assigned to gt_labels.shape[0]
    :param reliability_level: at average, diagonal elements of the confusion matrices for each crowd member (float)
    :param p_fill_task: a probability of each volunteer to perform a task, i.e. a float number in [0, 1]
    :return: labels - np.array(n_total_tasks, n_volunteers) is numpy nd-array of the size: number of tasks
    n_total_tasks, number of crowd members n_volunteers, where labels[i, j] is an answer for task i by volunteer j.
    If there is no answer, labels[i, j] = -1. Data points for which there is no labels from any of crowd members
    (if n_total_tasks > gt_labels.shape[0]) are concatenated to the end.
    """

    cm = generate_confusion_matrices_for_crowd_volunteers(n_volunteers, n_classes, reliability_level=reliability_level)
    labelled_crowdsourced_labels = generate_volunteer_answers_from_cm(cm, gt_labels, p_fill_task=p_fill_task)

    if n_total_tasks is None:
        n_total_tasks = gt_labels.shape[0]

    labels = expand_volunteer_labels(labelled_crowdsourced_labels, n_total_tasks)

    return labels
