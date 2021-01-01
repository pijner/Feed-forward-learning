from typing import List, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt


class AlphaSplayer:
    """
    Class to implement alpha splaying - takes in 1 dimensional training data and generates alpha values for given gamma
    range
    """

    def __init__(self, kernel: str, gamma: Union[np.ndarray, float, None] = None):
        """
        class initializer
        :param kernel: "linear" or "rbf" value for kernel type
        :param gamma: number of array of numbers for gamma parameter for rbf kernel
        """
        assert kernel in ["linear", "rbf"]
        self.kernel = kernel
        self.gamma_values = gamma

    def get_alpha(self, X: np.ndarray, y: np.ndarray, test: np.ndarray) -> np.ndarray:
        """
        function to get alpha values for test sample(s)
        :param X: (num_samples, num_features) shaped training array
        :param y: (num_samples, ) shaped binary training labels
        :param test: (num_samples, num_features) shaped testing array. Note that num_samples can be 1
        :return: (num_) array of alpha values
        """
        positive_term, negative_term, test_term = self.format_data(X, y, test)

        if self.kernel == "linear":
            pos, neg = self.get_terms_linear(positive_term, negative_term, test_term)
        else:
            pos, neg = self.get_terms_rbf(positive_term, negative_term, test_term)

        alpha_out = np.divide(neg, neg + pos)

        return alpha_out

    def alpha_for_all(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        function to get individual alpha values of all samples in the given set
        :param X: (num_samples, num_features) shaped array
        :param y: (num_samples, ) shaped binary labels
        :return:
        """
        all_alpha = []
        for i in range(np.shape(y)[0]):
            ts_X = X[i, :]
            tr_X = np.delete(X, i, axis=0)
            tr_y = np.delete(y, i)

            this_alpha = self.get_alpha(X=tr_X, y=tr_y, test=ts_X)
            try:
                all_alpha.append(this_alpha[0])
            except IndexError:
                all_alpha.append(this_alpha)

        return np.transpose(all_alpha)

    def format_data(self, X: np.ndarray, y: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        helper function to separate positive and negative class data and adjust array shapes to be broadcast-able with
        testing set
        :param X: (num_samples, num_features) shaped training array
        :param y: (num_samples, ) shaped binary training labels
        :param test: (num_test, num_features) or (num_features, ) shaped testing array
        :return: reshaped arrays,
            positive_term: (num_positive, num_features, 1) shaped array of positive class data
            negative_term: (num_negative, num_features, 1) shaped array of negative class data
            test_term: (1, num_features, num_test) shaped array of test data
        """
        assert np.ndim(X) == 2, "Expected 2 dimensional array of shape (num_records, num_attributes)"
        assert np.shape(X)[0] == np.shape(y)[0], "Number of records should match number of labels"

        label_classes = np.unique(y)
        assert label_classes.size == 2, "Expected binary classification data only"

        if np.ndim(test) == 1 and (np.shape(test)[0] == np.shape(X)[1]):  # if only one test sample was given
            test_term = np.expand_dims(test, (0, -1))
        elif np.ndim(test) == 2 and (np.shape(test)[1] == np.shape(X)[1]):  # if test has multiple samples
            test_term = np.expand_dims(np.swapaxes(test, 0, -1), 0)
        else:
            raise Exception(
                "test can either be a record (num_attributes, ) or a list of records (num_test_records, num_attributes)"
            )

        positive_term = np.expand_dims(X[y == label_classes[1], :], -1)
        negative_term = np.expand_dims(X[y == label_classes[0], :], -1)

        return positive_term, negative_term, test_term

    def predict(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test: np.ndarray,
        alpha: Union[np.ndarray, float],
        gamma: Union[np.ndarray, float, None] = None,
    ) -> np.ndarray:
        """
        function to predict class of testing data given training data and labels
        :param X: (num_samples, num_features) shaped training array
        :param y: (num_samples, ) shaped binary training labels
        :param test: (num_test, num_features) shaped testing array. Note that num_samples can be 1
        :param alpha: alpha value or array or alpha values (must have shape (num_gamma_values, )) for prediction
        :param gamma: gamma value or array of gamma values for RBF kernel in the final layer. Ignored if kernel is
            linear
        :return: array of predicted labels. If alpha or gamma is an array, shape is (num_test_samples, num_gamma_values)
            otherwise, shape is (num_test_samples)
        """

        positive_term, negative_term, test_term = self.format_data(X, y, test)

        # if any test sample has nan values, it's prediction will be -1
        test_nan = np.any(np.isnan(test_term[0, :, :]), axis=0)
        test_term = test_term[:, :, test_nan == 0]

        if self.kernel == "linear":
            pos, neg = self.get_terms_linear(positive_term, negative_term, test_term)
        else:
            pos, neg = self.get_terms_rbf(positive_term, negative_term, test_term, gamma=gamma)

        class_regression = alpha * pos - (1 - alpha) * neg
        class_regression = np.array(class_regression > 0, dtype=int)

        # insert predictions for nan values found above
        nan_indices = np.where(test_nan)[0]
        nan_indices[nan_indices >= np.shape(class_regression)[0]] = -1
        if np.ndim(class_regression) == 1:
            class_regression = np.insert(class_regression, nan_indices, -1)
        else:
            class_regression = np.insert(class_regression, nan_indices, -1, axis=0)

        return class_regression

    def get_terms_linear(
        self, positive_term: np.ndarray, negative_term: np.ndarray, test_term: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        function to get mean squared euclidean distance for each test term from positive and negative samples
        :param positive_term: (num_positive, num_features, 1) shaped positive class data from AlphaSplayer.format_data()
        :param negative_term: (num_negative, num_features, 1) shaped negative class data from AlphaSplayer.format_data()
        :param test_term: (1, num_features, num_test) shaped array of test data
        :return:
        """
        pos = (positive_term - test_term) ** 2
        pos = np.mean(pos, axis=0)
        pos = np.sum(1 - pos, axis=0)

        neg = (negative_term - test_term) ** 2
        neg = np.mean(neg, axis=0)
        neg = np.sum(1 - neg, axis=0)

        return pos, neg

    def get_terms_rbf(
        self,
        positive_term: np.ndarray,
        negative_term: np.ndarray,
        test_term: np.ndarray,
        gamma: Union[np.ndarray, float, None] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        function to calculate mean(e^(-1* gamma * squared euclidean distance)) for each test term from positive and
        negative samples
        :param positive_term: (num_positive, num_features, 1) shaped positive class data from AlphaSplayer.format_data()
        :param negative_term: (num_negative, num_features, 1) shaped negative class data from AlphaSplayer.format_data()
        :param test_term: (1, num_features, num_test) shaped array of test data
        :param gamma: number of array of numbers for gamma parameter for the RBF kernel
        :return:
        """
        if gamma is None:
            gamma = self.gamma_values

        pos = (positive_term - test_term) ** 2
        pos = np.sum(pos, axis=1) ** 1
        pos = np.exp(-1 * gamma * np.expand_dims(pos, -1))  # type: ignore

        neg = (negative_term - test_term) ** 2
        neg = np.sum(neg, axis=1) ** 1
        neg = np.exp(-1 * gamma * np.expand_dims(neg, -1))  # type: ignore

        # calculate means
        pos = np.mean(pos, axis=0)
        neg = np.mean(neg, axis=0)

        return pos, neg


class StackedAlphaSplayer:
    """
    Class to implement a model with stacked AlphaSplayer layers
    """

    def __init__(self, n_layers: int = 0, gamma_range: Union[np.ndarray, None] = None) -> None:
        """
        class initializer
        :param n_layers: number of layers to initialize the model with
        :param gamma_range: array containing gamma values for the layers. All layers will be initialized with the same
            gamma range
        """
        self.num_layers = 0
        self.layers: List[AlphaSplayer] = []
        for i in range(n_layers):
            self.add_layer(gamma_range=gamma_range)
        self.input_values: List[np.ndarray] = []
        self.labels: Union[np.ndarray, List] = []
        self.mean_alpha = 0.0

    def add_layer(self, kernel: str = "rbf", gamma_range: Union[np.ndarray, None] = None) -> None:
        """
        function to add layer of AlphaSplayer to the model
        :param kernel: "rbf" or "linear" kernel for layer
        :param gamma_range: (num_gamma_values, ) shaped array of gamma values for rbf kernel
        """
        if kernel == "rbf":
            assert gamma_range is not None, "Must specify gamma range if kernel is rbf"
        this_layer = AlphaSplayer(kernel=kernel, gamma=gamma_range)
        self.layers.append(this_layer)
        self.num_layers += 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        function to sequentially generate alpha values for all the layers in the model. The purpose is to find the mean
        alpha value to be used in the prediction equation
        :param X: (num_samples, num_features) shaped training array
        :param y: (num_samples, ) shaped array of training labels
        """
        assert self.num_layers > 0, "Stack should have at least 1 layer. Use model.add_layer() to add a layer"

        self.labels = y
        self.input_values.append(X)

        alpha_vals = np.empty(0)
        for layer in self.layers:
            layer_X = self.input_values[-1]

            alpha_vals = layer.alpha_for_all(layer_X, y)

            if layer.kernel == "rbf":
                to_purge = np.any(np.isnan(alpha_vals), axis=1)
                layer.gamma_values = layer.gamma_values[np.bitwise_not(to_purge)]  # type: ignore
                alpha_vals = alpha_vals[np.bitwise_not(to_purge), :]

            self.input_values.append(np.transpose(alpha_vals))

        if np.ndim(alpha_vals) == 1:  # if there's only one alpha value per sample (linear kernel)
            self.mean_alpha = np.mean(alpha_vals[alpha_vals != 0.5])
        else:
            self.mean_alpha = np.mean(alpha_vals, axis=1)

    def predict(
        self, X, method: str = "vote", gamma: Union[float, None] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        function to get predictions or a list of predictions for the testing data
        :param X: (num_test, num_features) shaped testing array
        :param method: one of "vote", "gamma_select", "return_all", or "linear"
            "vote": used when the final layer is rbf. Predictions are generated for each gamma value in the final layer
                and the majority is passed
            "gamma_select": used when the final layer is rbf. Predictions are generated for the given gamma value. If no
                gamma value is given, gamma = 1/(var(input to final layer) * (number of test samples))
            "return_all": used when the final layer is rbf. Predictions are generated for each gamma value and all
                predictions along with the gamma values are returned
            "linear": used when the final layer is linear
        :param gamma: gamma value to be used when method is gamma_select
        :return: array of predictions for given data. The gamma values of the final layer are also returned if method is
            return_all
        """
        assert self.num_layers > 0, "Stack should have at least 1 layer. Use add_layer() to add a layer"
        assert len(self.input_values) > 0, "Call model.predict() before calling model.predict()"

        # layer_X is the input for given layer
        layer_X = X
        predictions = []
        for i, layer in enumerate(self.layers):
            is_final = i == (self.num_layers - 1)

            if not is_final:
                alpha_vals = layer.get_alpha(self.input_values[i], self.labels, layer_X)
                layer_X = alpha_vals
                continue

            elif (method == "vote") or (method == "gamma_select") or (method == "return_all"):
                assert layer.kernel == "rbf", "Final layer must be rfb if method is vote, gamma_select, or return_all"

                if method == "gamma_select" and (gamma is None):
                    gamma = 1 / (np.var(layer_X) * np.shape(layer_X)[1])

                predictions = layer.predict(
                    self.input_values[i], self.labels, layer_X, alpha=self.mean_alpha, gamma=gamma
                )

                if method == "vote":
                    predictions = (np.sum(predictions, axis=-1) > (np.shape(predictions)[-1] / 2)).astype(int)
                elif method == "return_all":
                    return predictions, self.layers[-1].gamma_values
                else:
                    predictions = predictions[:, 0]  # type: ignore

            elif method == "linear":
                assert (
                    layer.kernel == "linear"
                ), "Final layer must be linear kernel if prediction method is not vote or gamma select"
                predictions = layer.predict(self.input_values[i], self.labels, layer_X, alpha=self.mean_alpha)

        return predictions

    def plot_gamma_acc(self, y_true: np.ndarray, y_pred: np.ndarray, gamma_range, filepath: str, name: str) -> None:
        """
        function to plot accuracies for predictions over a range of gamma values
        :param y_true: (num_samples, ) shaped array of binary class labels
        :param y_pred: (num_samples, num_gamma_values) shaped array of predictions
        :param gamma_range: array of gamma value for which predictions are given
        :param filepath: filepath to save the figure (includes filename)
        :param name: title for the plot
        """
        accuracy = np.sum(np.equal(y_pred, np.expand_dims(y_true, -1)), axis=0) / np.shape(y_true)[0]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(gamma_range, accuracy)
        ax.set_xlabel("Gamma")
        ax.set_ylabel("Accuracy")
        plt.title("Gamma v/s Accuracy for " + name)
        plt.savefig(filepath)
