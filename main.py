from math import isqrt

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def read_data(file, delimiter=None):
    data = np.loadtxt(file, delimiter=delimiter)
    X = data[:, 1:]
    y = data[:, 0]
    return X, y


X_train, y_train = read_data('data/train.txt')
X_test, y_test = read_data('data/test.txt')


def show_sample(X, index):
    """Takes a dataset (e.g. X_train) and imshows the digit at the corresponding index

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        index (int): index of digit to show
    """

    # 1d array to square 2d array
    d = isqrt(X.shape[1])
    img = X[index].reshape(d, d)

    plt.imshow(img, cmap='Greys')
    plt.axis('off')
    plt.show()


def plot_digits_samples(X, y):
    """Takes a dataset and selects one example from each label and plots it in subplots

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
    """

    # Find the unique labels and a corresponding image for each label
    d = isqrt(X.shape[1])  # Dimension of the d x d image.
    labels, indices = np.unique(y, return_index=True)
    images = list(zip(X[indices].reshape(-1, d, d), labels))  # List of image, label pairs
    images.sort(key=lambda v: v[1])  # Sort by label

    # Calculate the dimensions of the image grid which we will display
    n = len(images)
    k = isqrt(n)
    q, r = divmod(n - k * k, k)
    ncols = k
    nrows = k + q + (r != 0)  # ceiling division

    # Plot the images
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 12))
    for i, (img, label) in enumerate(images):
        ax = axs.flat[i]
        ax.imshow(img, cmap='Greys')
        ax.set_title(label, size='x-large')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    # Strip the axes of the unfilled parts of the grid (if there are any)
    for ax in axs.flat[len(images):]:
        ax.axis('off')

    fig.suptitle('Labeled Images', size='xx-large')

    plt.show()


def digit_mean_at_pixel(X, y, digit, pixel=(10, 10)):
    """Calculates the mean for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixel (tuple of ints): The pixels we need to select.

    Returns:
        (float): The mean value of the digits for the specified pixels
    """

    d = isqrt(X.shape[1])  # d by d image
    Xd = X[y == digit]  # Select the digits
    Xd = Xd.reshape(-1, d, d)  # Reshape for easier indexing
    Xdp = Xd[:, pixel[0], pixel[1]]  # Select the pixel

    return np.mean(Xdp)


# # Test
# img = np.array([[digit_mean_at_pixel(X_test, y_test, 8, pixel=(i,j)) for j in range(16)]
#                for i in range(16)])
# plt.imshow(img)
# plt.show()


def digit_variance_at_pixel(X, y, digit, pixel=(10, 10)):
    """Calculates the variance for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixel (tuple of ints): The pixels we need to select

    Returns:
        (float): The variance value of the digits for the specified pixels
    """

    d = isqrt(X.shape[1])  # d by d image
    Xd = X[y == digit]  # Select the digits
    Xd = Xd.reshape(-1, d, d)  # Reshape for easier indexing
    Xdp = Xd[:, pixel[0], pixel[1]]  # Select the pixel

    return np.var(Xdp.var)


# # Test
# assert X_test[y_test==7].reshape(-1, 16, 16)[:, 15, 15].var() == 0


def digit_mean(X, y, digit):
    """Calculates the mean for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The mean value of the digits for every pixel
    """
    return np.mean(X[y == digit], axis=0)


def digit_variance(X, y, digit):
    """Calculates the variance for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The variance value of the digits for every pixel
    """
    return np.var(X[y == digit], axis=0)


# t = np.vstack([digit_variance(X_test, y_test, digit)
#                for digit in np.unique(y_test)])


def euclidean_distance(s, m):
    """Calculates the euclidean distance between a sample s and a mean template m

    Args:
        s (np.ndarray): Sample (nfeatures)
        m (np.ndarray): Template (nfeatures)

    Returns:
        (float) The Euclidean distance between s and m
    """
    delta = s - m
    return (delta @ delta) ** 0.5


# m = digit_mean(X_test, y_test, 3)
# for i in range(100):
#     print(int(y_test[i]), euclidean_distance(X_test[i], m))


def euclidean_distance_classifier(X, X_mean):
    """Classifiece based on the euclidean distance between samples in X and template vectors in X_mean

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        X_mean (np.ndarray): Digits data (n_classes x nfeatures)

    Returns:
        (np.ndarray) predictions (nsamples)
    """

    # We add a new dimension in the middle of the means array so that
    # X      :               nsamples, nfeatures
    # X_means:     nclasses,        1, nfeatures
    # and the resulting shape will be (nclasses, nsamples, nfeatures)
    # where the [i, j, :] element will represent  sample[j] - class[i]
    # We then take the L2 norm square along the final dimension (the features)
    # And then take the argmin along the first dimension (the classes)

    X_mean = np.expand_dims(X_mean, axis=1)
    diffs = X - X_mean
    dists2 = np.einsum('ijk, ijk -> ij', diffs, diffs)
    y_pred = np.argmin(dists2, axis=0)

    return y_pred


# # Compute all means
# means = np.vstack([digit_mean(X_train, y_train, digit) for digit in range(10)])
# # Predict and compute accuracy
# y_pred = euclidean_distance_classifier(X_test, means)
# print(np.mean(y_pred == y_test))


class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):
    """Classify samples based on the distance from the mean feature value"""

    # https://scikit-learn.org/stable/developers/develop.html

    # __init__ should only be setting data independent parameters (mainly hyperparameters)
    # def __init__(self):
    #     self.X_mean_ = None

    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        fit always returns self.
        """

        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_means_ = np.vstack([np.mean(X[y == c], axis=0) for c in self.classes_])

        return self

    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        check_is_fitted(self)
        X = check_array(X)
        indices = euclidean_distance_classifier(X, self.X_means_)
        return self.classes_[indices]

    # # ClassifierMixin automatically implements this
    # def score(self, X, y):
    #     """
    #     Return accuracy score on the predictions
    #     for X based on ground truth y
    #     """
    #     y_pred = self.predict(X)
    #     return np.mean(y == y_pred)


# clf = EuclideanDistanceClassifier()
# clf.fit(X_train, y_train)
# score = clf.score(X_test, y_test)
# print(score)


def evaluate_classifier(clf, X, y, n_folds=5):
    """Returns the 5-fold accuracy for classifier clf on X and y

    Args:
        clf (sklearn.base.BaseEstimator): classifier
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        n_folds (int): Number of folds for the cross validation for the score.

    Returns:
        (float): The 5-fold classification score (accuracy)
    """
    scores = cross_val_score(clf, X, y, cv=n_folds, scoring='accuracy')
    return np.mean(scores)


# clf = EuclideanDistanceClassifier()
# score = evaluate_classifier(clf, X_train, y_train)
# print(score)


def calculate_priors(y):
    # Removed X from the arguments since it's not relevant in the computation
    """Return the a-priori probabilities for every class

    Args:
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (np.ndarray): (n_classes) Prior probabilities for every class
    """
    _, freqs = np.unique(y, return_counts=True)
    return freqs / len(y)


def plot_learning_curve(estimator, X, y,
                        title='Learning Curves', ax=None,
                        cv=5, train_sizes=np.linspace(0., 1., 11)[1:]):
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    # --- Computational part ---
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y,
                                                            train_sizes=train_sizes, cv=cv)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # --- Plotting part ---
    if ax is None:
        _, ax = plt.subplots()

    # Plot the curves
    ax.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    ax.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')

    # Plot +-1 accuracy standard deviation strips
    ax.fill_between(train_sizes,
                    train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std,
                    alpha=0.1,
                    color='r')
    ax.fill_between(train_sizes,
                    test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std,
                    alpha=0.1,
                    color='g')

    # Decorate
    ax.set_title(title)
    ax.set_xlabel('Training set size')
    ax.set_ylabel('Estimator score')
    plt.legend(loc='best')
    ax.grid()

    return ax


# clf = EuclideanDistanceClassifier()
# plot_learning_curve(clf, X_train, y_train,
#                     title='Learning Curves (Euclidean Classifier)')
# plt.show()


class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""

    def __init__(self, use_unit_variance=False, priors=None, var_smoothing=1e-9):
        self.use_unit_variance = use_unit_variance
        self.priors = priors
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        fit always returns self.
        """

        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        # p(Ci) ~ empirical distribution of the classes
        self.priors_ = calculate_priors(y) if self.priors is None else np.asarray(self.priors)

        grouped = [X[y == c] for c in self.classes_]
        # Estimation of the mean and variance of the likelihood p(xj|Ci) ~ N(mu_ij, sigma_ij)
        self.mus_ = np.vstack([np.mean(g, axis=0) for g in grouped])
        if self.use_unit_variance:
            self.sigmas_ = np.ones((np.size(self.classes_), X[0].shape[0]))
        else:
            self.sigmas_ = np.vstack([np.var(g, axis=0) for g in grouped])

        self.epsilon_ = np.max(self.sigmas_, axis=None) * self.var_smoothing

        return self

    def _losses(self, X):
        # The class which maximizes the likelihood will be equal to np.argmin(self._losses(X))
        # The evidence is not included since it's irrelevant in calculating the argmax class.
        # In *Naive* Bayes we assume independence of the features when the class is given.
        # We will also assume that the likelihood is normally distributed. (GAUSSIAN Naive Bayes)
        X = np.expand_dims(X, axis=1)
        sigmas = self.sigmas_ + self.epsilon_
        log_priors = np.log(self.priors_)
        sum_log_sigmas = np.sum(np.log(sigmas), axis=1)
        sum_squares = np.sum((X - self.mus_)**2 / sigmas, axis=-1)
        likelihood_equivalent = log_priors - 0.5 * sum_log_sigmas - 0.5 * sum_squares
        return -likelihood_equivalent

    def predict_log_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        likelihood_equivalent = -self._losses(X)
        log_normalizer = logsumexp(likelihood_equivalent, axis=1)
        log_normalizer = np.expand_dims(log_normalizer, axis=1)
        return likelihood_equivalent - log_normalizer

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        check_is_fitted(self)
        X = check_array(X)
        losses = self._losses(X)
        indices = np.argmin(losses, axis=1)
        return self.classes_[indices]

    # # ClassifierMixin automatically implements this
    # def score(self, X, y):
    #     """
    #     Return accuracy score on the predictions
    #     for X based on ground truth y
    #     """
    #     y_pred = self.predict(X)
    #     return np.mean(y == y_pred)



# def train_eval(estimator):
#     estimator.fit(X_train, y_train)
#     return estimator.score(X_test, y_test)
#
#
# score_custom = train_eval(CustomNBClassifier())
#
# from sklearn.naive_bayes import GaussianNB
# score_sklearn = train_eval(GaussianNB())
#
# score_unit_var = train_eval(CustomNBClassifier(use_unit_variance=True))
#
# n_classes = np.size(np.unique(y_train))
# uniform_prior = np.ones(n_classes) / n_classes
# score_unit_var_uniform_priors = train_eval(CustomNBClassifier(priors=uniform_prior, use_unit_variance=True))
#
# score_eucl = train_eval(EuclideanDistanceClassifier())
#
# print(f'''{score_custom = }
# {score_sklearn = }
# {score_unit_var = }
# {score_unit_var_uniform_priors = }
# {score_eucl = }''')


# clf1 = CustomNBClassifier()
# clf1.fit(X_train, y_train)
# log_proba1 = clf1.predict_log_proba(X_test)
# proba1 = clf1.predict_proba(X_test)
# pred1 = clf1.predict(X_test)
# score1 = clf1.score(X_test, y_test)
#
# clf2 = GaussianNB()
# clf2.fit(X_train, y_train)
# log_proba2 = clf2.predict_log_proba(X_test)
# proba2 = clf2.predict_proba(X_test)
# pred2 = clf2.predict(X_test)
# score2 = clf2.score(X_test, y_test)
#
# clf = CustomNBClassifier()
# clf.fit(X_train, y_train)
# mus = clf.mus_
# diffs = np.expand_dims(mus, 1) - mus  # (10, 10, 256)
# dists_squared = np.einsum('...j, ...j', diffs, diffs)  # (10, 10)

