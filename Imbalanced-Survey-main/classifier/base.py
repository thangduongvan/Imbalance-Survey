from scipy.sparse import data
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from collections import Counter
import datetime
import numpy as np



class SamplingClassifier(BaseEstimator, ClassifierMixin):
    """
    This class wraps an sampler and a classifier, making it compatible
    with sklearn based pipelines.
    """

    def __init__(self, sampler, classifier, dataset_name='Default'):
        """
        Constructor of the wrapper.

        Args:
            sampler (obj): an sampler object (if is None, only run classifier)
            classifier (obj): an sklearn-compatible classifier
            dataset_name (str): Dataset name (optinal)
        """

        self.sampler = sampler
        self.classifier = classifier
        self.dataset_name = dataset_name

    def fit(self, X, y=None):
        """
        Carries out sampling and fits the classifier.

        Args:
            X (np.ndarray): feature vectors
            y (np.array): target values

        Returns:
            obj: the object itself
        """
        X_ = X
        y_ = y
        self.y_orig = y
        if self.sampler is not None:
            try:
                X_, y_ = self.sampler.sample(X, y)
            except Exception as e1:
                # compatibility with imbalanced-learn pipelines
                try:
                    X_, y_ = self.sampler.fit_resample(X, y)
                except Exception as e2:
                    self.y_samp = y_
                    with open('./log_error_base.txt', 'a') as f:
                        f.write(f'{datetime.datetime.now()}\t{self.get_params()}\t{e1}\t{e2}\n')
        self.y_samp = y_
        # logging.info("\tValue counts: {0}\n".format(Counter(y_)))
        self.classifier.fit(X_, y_)

        return self

    def predict(self, X):
        """
        Carries out the predictions.

        Args:
            X (np.ndarray): feature vectors
        """

        return self.classifier.predict(X)

    def predict_proba(self, X):
        """
        Carries out the predictions with probability estimations.

        Args:
            X (np.ndarray): feature vectors
        """

        return self.classifier.predict_proba(X)

    def get_params(self, deep=True):
        """
        Returns the dictionary of parameters.

        Args:
            deep (bool): wether to return parameters with deep discovery

        Returns:
            dict: the dictionary of parameters
        """
        return {
            'sampler': self.sampler.__class__.__name__,
            'classifier': self.classifier.__class__.__name__,
            'dataset_name': self.dataset_name,
            'y_orig_counts': Counter(self.y_orig),
            'y_orig_counts': Counter(self.y_samp),
        }

    def set_params(self, **parameters):
        """
        Sets the parameters.

        Args:
            parameters (dict): the parameters to set.

        Returns:
            obj: the object itself
        """

        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self
