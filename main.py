import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import zscore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.AlphaSplayer import StackedAlphaSplayer
from src.get_data import get_all

GAMMA_VALUES = np.sort(np.append(np.linspace(0.1, 3, 100), np.logspace(-5, -1, 100)))


def profile_stas(X: np.ndarray, y: np.ndarray, n_layers: int = 2) -> None:
    """
    function to make plots to observe the alpha values at each layer and plot final layer gamma v/s accuracy for a
    StackedAlphaSplayer. Uses a 80-20 train-test split on the given data
    :param X: (num_samples, num_features) shaped data array
    :param y: (num_samples, ) shaped label array
    :param n_layers: number of layers in StackedAlphaSplayer
    """
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0, train_size=0.8)

    sModel = StackedAlphaSplayer(n_layers=n_layers, gamma_range=GAMMA_VALUES)
    sModel.fit(X=train_X, y=train_y)

    class_0 = np.where(train_y == 0)[0]
    class_1 = np.where(train_y == 1)[0]
    plot_alpha_profile(sModel, [class_0, class_1])

    pred, final_gamma_range = sModel.predict(X=test_X, method="return_all")
    sModel.plot_gamma_acc(
        y_true=test_y,
        y_pred=pred,
        gamma_range=final_gamma_range,
        name=dataset_name,
        filepath=str(Path("./figs/gamma_accuracies/").joinpath(dataset_name + "_" + str(n_layers))),
    )


def plot_alpha_profile(model: StackedAlphaSplayer, class_indices: List[np.ndarray]) -> None:
    """
    function to plot the layer wise alpha values of each observation with respect to the gamma values
    :param model: StackedAlphaSplayer model whose alpha values are to be plotted
    :param class_indices: list of 2 arrays, first containing indices of class 0, second containing indices of class 1
    """
    cols = model.num_layers
    fig, ax = plt.subplots(nrows=2, ncols=cols, figsize=(5 * cols, 12), subplot_kw={"projection": Axes3D.name})

    colors = ["tab:blue", "tab:orange"]
    for i in range(cols):
        for j in range(2):      # we only have 2 classes
            a = np.ravel([model.layers[i].gamma_values] * np.shape(class_indices[j])[0])
            b = np.ravel(model.input_values[i + 1][class_indices[j], :])
            c = np.repeat(class_indices[j], np.shape(model.input_values[i + 1])[1])

            if cols > 1:        # if the number of layers is more than 1
                ax[j, i].scatter(c, a, b, marker=".", alpha=0.4, c=colors[j])

                ax[j, i].set_xlabel("Sample number")
                ax[j, i].set_ylabel("Gamma")
                ax[j, i].set_zlabel("Alpha")

                ax[j, i].set_title("Layer {}, Class {}".format(i + 1, j))

            else:               # if the number of layers is 1
                ax[j].scatter(c, a, b, marker=".", alpha=0.4, c=colors[j])

                ax[j].set_xlabel("Sample number")
                ax[j].set_ylabel("Gamma")
                ax[j].set_zlabel("Alpha")

                ax[j].set_title("Layer {}, Class {}".format(i + 1, j))

    plt.suptitle("Gamma v/s alpha for " + dataset_name)
    plt.tight_layout()
    plt.savefig(str(Path("./figs/alpha_profiles/").joinpath(dataset_name)))


def validate_learners(X: np.ndarray, y: np.ndarray) -> List[str]:
    """
    function to cross validate all three of the prediction methods of StackedAlphaSplayer and some default valued scikit
    learn classifiers. The test statistics are saved in a list of strings to be written to a file
    :param X: (num_samples, num_features) shaped data
    :param y: (num_samples, ) shaped labels
    :return: list of strings containing test stats to be written to a file
    """
    accuracies: Dict[str, List] = {}
    auc: Dict[str, List] = {}

    k_fold_splits = StratifiedKFold(n_splits=5).split(X, y)

    for train_idx, test_idx in k_fold_splits:
        train_X, test_X = X[train_idx, :], X[test_idx, :]
        train_y, test_y = y[train_idx], y[test_idx]

        classifiers = {
            "svm_lin": SVC(kernel="linear"),
            "svm_rbf": SVC(kernel="rbf"),
            "rf_1": RandomForestClassifier(n_estimators=1, n_jobs=-1),
            "rf_10": RandomForestClassifier(n_estimators=10, n_jobs=-1),
            "mlp": MLPClassifier(hidden_layer_sizes=(32, 64, 32)),
            "stas_vote_1": StackedAlphaSplayer(n_layers=1, gamma_range=GAMMA_VALUES),
            "stas_gamma_select_1": StackedAlphaSplayer(n_layers=1, gamma_range=GAMMA_VALUES),
            "stas_linear_1": StackedAlphaSplayer(n_layers=1, gamma_range=GAMMA_VALUES),
            "stas_vote_2": StackedAlphaSplayer(n_layers=2, gamma_range=GAMMA_VALUES),
            "stas_gamma_select_2": StackedAlphaSplayer(n_layers=2, gamma_range=GAMMA_VALUES),
            "stas_linear_2": StackedAlphaSplayer(n_layers=2, gamma_range=GAMMA_VALUES),
        }

        # Final layer needs to be a linear kernel for this prediction method
        classifiers["stas_linear_1"].add_layer(kernel="linear")
        classifiers["stas_linear_2"].add_layer(kernel="linear")

        for c in classifiers.keys():
            classifiers[c].fit(X=train_X, y=train_y)
            if c in ["stas_gamma_select_1", "stas_gamma_select_2"]:
                predictions = classifiers[c].predict(X=test_X, method="gamma_select")
            elif c in ["stas_linear_1", "stas_linear_2"]:
                predictions = classifiers[c].predict(X=test_X, method="linear")
            else:
                predictions = classifiers[c].predict(X=test_X)

            this_accuracy = np.sum(np.equal(predictions, test_y)) / np.shape(test_y)[0]
            this_auc = roc_auc_score(y_true=test_y, y_score=predictions)

            if c in accuracies.keys():
                accuracies[c].append(this_accuracy)
                auc[c].append(this_auc)
            else:
                accuracies[c] = [this_accuracy]
                auc[c] = [this_auc]

    df_acc = pd.DataFrame(accuracies)
    df_auc = pd.DataFrame(auc)

    contents = []
    means = pd.DataFrame({"mean_acc": df_acc.mean(), "mean_auc": df_auc.mean()}).sort_values(
        "mean_acc", ascending=False
    )

    contents.append("Stats for dataset " + dataset_name + "\n")
    contents.append(means.to_markdown())
    contents.append("\n")
    contents.append("-" * 50)
    contents.append("\n\n")

    return contents


def dataset_description(X: np.ndarray, y: np.ndarray, name: str = "") -> List[str]:
    """
    function to get dataset shape and leading AUC values. All the stats are dumped in a list of strings and returned
    :param X: (num_samples, num_features) shaped data
    :param y: (num_samples, ) shaped labels
    :param name: Name of the dataset
    :return: list of strings containing dataset information to be written to a file
    """
    contents = [
        "### Dataset: " + name,
        "\nNumber of attributes: {}<br>\n".format(np.shape(X)[1]),
        "Number of records: {}\n".format(np.shape(X)[0]),
    ]
    classes = np.unique(y)

    group_0 = np.where(y == classes[0])[0]
    group_1 = np.where(y == classes[1])[0]

    contents.append("Num Class 0: {}\n".format(np.shape(group_0)[0]))
    contents.append("Num Class 1: {}\n".format(np.shape(group_1)[0]))

    feature_data = np.concatenate((X[group_0, :], X[group_1, :]))
    feature_labels = np.concatenate((np.zeros_like(group_0), np.ones_like(group_1)))

    roc_values = np.array([roc_auc_score(feature_labels, feature_data[:, i]) for i in range(np.shape(feature_data)[1])])
    best_measurements = np.argsort(-1 * np.abs(roc_values - 0.5))

    auc_table = pd.DataFrame(
        columns=["Feature", "AUC"],
        data=np.transpose([["Feature #%d" % i for i in best_measurements], roc_values[best_measurements]]),
    )

    auc_table = auc_table.set_index("Feature").astype({"AUC": float})
    contents.append("\nAUC Values\n")
    if np.shape(X)[1] > 10:
        contents.append(auc_table.iloc[:10, :].round(5).to_markdown())
    else:
        contents.append(auc_table.round(5).to_markdown())

    contents.append("\n")
    contents.append("-" * 50)
    contents.append("\n\n")

    return contents


def write_file(contents: List[str], filepath: str) -> None:
    """
    helper function to write stats to files
    :param contents: list of strings to be written to files
    :param filepath: path of file to be written to (include filename and extension)
    """
    f = open(filepath, "w")
    f.writelines(contents)
    f.close()


if __name__ == "__main__":
    # read all data
    datasets = get_all()

    # initialize lists to be written to stats files
    descriptions: List[str] = []
    test_stats: List[str] = []

    for dataset_name in datasets.keys():
        print("Working on dataset ", dataset_name)
        data, labels = datasets[dataset_name]

        descriptions = descriptions + dataset_description(data, labels, dataset_name)

        z_data = zscore(data)
        z_valid = np.bitwise_not(np.any(np.isnan(z_data), axis=0))
        data[:, z_valid] = z_data[:, z_valid]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            profile_stas(data, labels, n_layers=1)
            profile_stas(data, labels, n_layers=2)
            test_stats = test_stats + validate_learners(data, labels)

    # write dataset description and test statistic files to ./stats/
    write_file(descriptions, str(Path("./stats/descriptions")) + ".md")
    write_file(test_stats, str(Path("./stats/test_stats")) + ".md")
