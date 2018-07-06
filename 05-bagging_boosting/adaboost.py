import numpy as np
from decisionstump import DecisionStump
from itertools import count

class Adaboost:
    def __init__(self, weak_learner_class=DecisionStump, weak_learners=None, learner_weights=None):

        self.WeakLearner = weak_learner_class
        self.weak_learners = weak_learners if weak_learners is not None else []
        self.learner_weights = learner_weights if learner_weights is not None else []


    def train( self, X, Y, num_weak_learners ):
        """
        Trains an Adaboost classifier
        :param X: NxD matrix, N observations, d props
        :param Y: Nx1 vector, ground truth
        :param num_weak_learners: scalar, num of weak classifiers to use as a base
        """

        n,d = X.shape

        # Init the weighting coeff
        sample_weights = np.ones(n)/n

        for m in range(num_weak_learners):
            print("Fitting weak classifier #%d" % m)

            weak_learner = self.WeakLearner()
            weak_learner.fit( X, Y, sample_weights )
            weak_predictions = weak_learner.predict(X)

            idx_miss = np.not_equal(weak_predictions, Y)
            err = np.dot(sample_weights, idx_miss) / sum(sample_weights)
            err = max(1e-8, err) # avoid zero division
            alpha = np.log((1-err)/err)
            self.weak_learners.append(weak_learner)
            self.learner_weights.append(alpha)

            # Update the data weighting coeff for next learner
            if m == num_weak_learners -1 :
                break

            sample_weights = [sw*np.exp(alpha*im) for sw, im in zip(sample_weights, idx_miss)]
            sample_weights /= sum(sample_weights)

    def add_learner(self, X, Y):
        """
        Add a weak learner to an existing AdaBoost classifier
        :param X: NxD
        :param Y: Nx1 ground truth
        """
        n, d = X.shape

        sample_weights = np.ones(n)/n

        m = count()

        for weak_learner, learner_weight in zip(self.weak_learners, self.learner_weights):
            print('Evaluating weak classifier #%d' % m.__next__())
            new_weak_learner = self.WeakLearner()
            new_weak_learner.fit(X, Y, sample_weights)
            weak_predictions = new_weak_learner.predict(X)
            idx_miss = np.not_equal(weak_predictions, Y)

            # evaluate quantities epsilon_m and alpha_m
            err = np.dot(sample_weights, idx_miss) / sum(sample_weights)
            err = max(1e-8, err)
            alpha = np.log((1-err)/err)

            sample_weights = [sw*np.exp(alpha*im) for sw, im in zip(sample_weights, idx_miss)]
            sample_weights /= sum(sample_weights)

        print('Fitting weak classifier #%d' % m.__next__())
        new_weak_learner = self.WeakLearner()
        new_weak_learner.fit( X, Y, sample_weights )
        weak_predictions = new_weak_learner.predict(X)
        idx_miss = np.not_equal(weak_predictions, Y)

        err = np.dot( sample_weights, idx_miss ) / sum(sample_weights)
        err = max(1e-8, err)
        alpha = np.log((1-err)/err)
        self.weak_learners.append(new_weak_learner)
        self.learner_weights.append(alpha)

    def predict(self, X):
        n, d = X.shape
        predictions = np.zeros(n)

        for weak_learner, learner_weight in zip(self.weak_learners, self.learner_weights):
            predictions_weak = weak_learner.predict(X)
            predictions += learner_weight*predictions_weak

        predictions = np.array( [1 if p>0 else -1 for p in predictions])
        return predictions


    def prediction_error(self, X,Y):
        predictions = self.predict(X)
        err = np.mean(np.not_equal(predictions, Y))
        return err
