# Topic: Bagging and Boosting

## Exercise 1: Bootstrag Aggregation

1. Q: What is the core idea in bagging? How does it differ from boosting?
    - Bagging 
        - Is a meta-algorithm to improve the accuracy and stability
        - randomly sample points with replacement from a training set
        - repeated several times to generate M different models/classifiers
        - predictions of different models averaged

    - Boosting
        - use all samples of the training set in each model exactly once
        - stack models on top of each other
        - data points are weighted by each model according to their misclassification rate according to the previous model in the stack
        - focus on harder samples
        - meta-estimator weights each model according to their final training accuracy

2. Does bagging reduce the bias of predictions, the variation or both?
    - Bagging reduces variance by averaging over learned models
    - bias might be even increased, but the variance reduction is relatively larger and therefore the total error is reduced

3. What is the out-of-bag error and why is it useful?
    - OOB error is calculated on unused data points (not used during training)
    - points have not been seen by the model
    - perfect for testing the generalization powers of the current model
    - alternative to cross-validation
    - only usable with bagging


## Exercise 2: Adaboost (Programming)
Check Jupyter Notebook for this
