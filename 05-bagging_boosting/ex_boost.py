# MLCV Winter 2016
#   Homework Assignment 04 - Boosting
#
#   Exercise 2) Adaboost
#
###########################################################################

import numpy as np
import random
from math import ceil
import matplotlib.pyplot as plt
from adaboost import AdaBoost
from decisionstump import DecisionStump


def split_dataset(targets, ratio):
    n = len(targets)
    labels = list(set(targets))
    splitting = np.zeros(n, dtype=bool)
    for label in labels:
        idx = [i for i,t in enumerate(targets) if t==label]
        num_left = ceil(len(idx)*ratio)
        idx_left = random.sample(idx, num_left)
        splitting[idx_left] = True
    return splitting


def plot_results(X, Y, idx_train, train_error, test_error):
    # Displaying the training and testing sets
    n = len(Y)
    plt.figure()
    num_rows, num_cols, sp = 2, 2, 1
    for row in range(num_rows):
        idx_set = idx_train if row == 0 else ~idx_train
        plt.subplot(num_rows, num_cols, row*num_cols + sp)
        idx_pos = [i for i in range(n) if idx_set[i] and Y[i]==1]
        idx_neg = [i for i in range(n) if idx_set[i] and Y[i]!=1]
        plt.scatter(X[idx_pos, 0], X[idx_pos, 1], c='b', lw=0, s=50)
        plt.scatter(X[idx_neg, 0], X[idx_neg, 1], c='r', lw=0, s=50)
        plt.title('Training set') if row == 0 else plt.title('Testing set')
        plt.grid(True)

        plt.subplot(num_rows, num_cols, row*num_cols + sp + 1)
        plt.plot(train_error) if row==0 else plt.plot(test_error)
        plt.axis([1, len(train_error), 0, 1])
        plt.title('Training Error')  if row==0 else plt.title('Testing error')
        plt.xlabel('#weak classifiers')
        plt.ylabel('error rate')
        plt.grid(True)
    plt.show()


def main():
    # Read in the dataset
    X = np.loadtxt('banknote_auth/banknote_auth_data.csv', delimiter=',')
    Y = np.loadtxt('banknote_auth/banknote_auth_labels.csv', dtype=str)

    # Map labels to -1 or +1
    labels = list(set(Y))
    Y = np.array([-1 if y==labels[0] else 1 for y in Y])

    ## Creating the training and testing sets
    training_ratio = 0.1
    n = len(Y)
    perm = np.random.permutation(n)
    X, Y = X[perm, :], Y[perm]
    idx_train = split_dataset(Y, training_ratio)

    ## Adaboost
    max_num_weak_learners = 40

    # Training and testing error rates
    train_error = [np.inf]
    test_error = [np.inf]
    model = AdaBoost(weak_learner_class=DecisionStump)
    for num_weak_learners in range(1, max_num_weak_learners+1):
        print('Training AdaBoost with %d weak learners...' % num_weak_learners)
        # if num_weak_learners == 1:
        #     model.train(X[idx_train, :], Y[idx_train], num_weak_learners)
        # else:
        model.add_learner(X[idx_train, :], Y[idx_train])

        train_error.append(model.prediction_error(X[idx_train, :], Y[idx_train]))
        test_error.append(model.prediction_error(X[~idx_train, :], Y[~idx_train]))

    print('Initial training error=%.4f, test error=%.4f' % (train_error[1], test_error[1]))
    print('Final training error=%.4f, test error=%.4f' % (train_error[-1], test_error[-1]))
    plot_results(X, Y, idx_train, train_error, test_error)


if __name__ == '__main__':
    main()