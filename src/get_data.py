"""
This file contains helper functions to read datasets in ./data/
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def get_arrhythmia(data_dir: Path = Path("./data/Arrhythmia")):
    clean_path = str(data_dir.joinpath("arrhythmia.data"))
    data = np.loadtxt(clean_path, delimiter=",", dtype=str)
    # Remove columns with missing data
    cols_to_keep = np.all(data != "?", axis=0)
    data = (data[:, cols_to_keep]).astype(float)

    # Last column represents the labels
    labels = np.array(data[:, -1] != 1, dtype=int)
    data = np.delete(data, -1, axis=1)
    return data, labels


def get_banknote(data_dir: Path = Path("./data/Bank_Note")):
    clean_path = str(data_dir.joinpath("data_banknote_authentication.txt"))
    data = np.loadtxt(clean_path, delimiter=",")

    # Last column represents the labels
    return data[:, :-1], data[:, -1].astype(int)


def get_bloodtransf(data_dir: Path = Path("./data/Blood_Transfusion")):
    clean_path = str(data_dir.joinpath("transfusion.data"))
    data = np.loadtxt(clean_path, delimiter=",", skiprows=1)

    # Last column represents the labels
    return data[:, :-1], data[:, -1]


def get_breastcancer(data_dir: Path = Path("./data/Breast_Cancer")):
    clean_path = str(data_dir.joinpath("wdbc.data"))
    data = np.loadtxt(clean_path, delimiter=",", dtype=str)

    # Second column represents labels (M = malignant, B = benign)
    labels = (data[:, 1] == "M").astype(int)
    data = (np.delete(data, [0, 1], axis=1)).astype(float)

    return data, labels


def get_ilpd(data_dir: Path = Path("./data/ILPD")):
    clean_path = str(data_dir.joinpath("Indian Liver Patient Dataset (ILPD).csv"))
    data = np.loadtxt(clean_path, delimiter=",", dtype=str)

    # Binarize gender
    data[:, 1] = 1 * (data[:, 1] == "Female")
    # Clear records with missing values
    data = np.delete(data, np.where(np.any(data == "", axis=1))[0], axis=0)
    data = data.astype(float)

    labels = np.array(data[:, -1] == 1, dtype=int)
    data = np.delete(data, -1, axis=1)

    return data, labels


def get_ionosphere(data_dir: Path = Path("./data/Ionosphere")):
    clean_path = str(data_dir.joinpath("ionosphere.data"))
    data = np.loadtxt(clean_path, delimiter=",", dtype=str)

    # Last column represents labels (g = good, b = bad)
    labels = (data[:, -1] == "g").astype(int)
    data = (np.delete(data, -1, axis=1)).astype(float)

    return data, labels


def get_parkinsons(data_dir: Path = Path("./data/PD")):
    clean_path = str(data_dir.joinpath("parkinsons.data"))
    data = np.loadtxt(clean_path, delimiter=",", dtype=str, skiprows=1)

    # Delete attribute name
    data = (np.delete(data, 0, axis=1)).astype(float)

    # Column 16 represents status
    labels = data[:, 16]
    data = np.delete(data, 16, axis=1)

    return data, labels


def get_pima(data_dir: Path = Path("./data/PIMA")):
    clean_path = str(data_dir.joinpath("diabetes.csv"))
    data = np.loadtxt(clean_path, delimiter=",", skiprows=1)

    labels = data[:, -1]
    data = np.delete(data, -1, axis=1)

    return data, labels


def get_wine(data_dir: Path = Path("./data/Wine")):
    clean_path = str(data_dir.joinpath("wine.data"))
    data = np.loadtxt(clean_path, delimiter=",")

    # We're only interested in binary classification so we'll pick out class 1 and 2
    data = data[data[:, 0] != 3, :]

    labels = np.array(data[:, 0] == 1, dtype=int)
    data = np.delete(data, 0, axis=1)

    return data, labels


def get_all() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Function to return a dictionary containing all datasets
    """
    datasets = {
        "arrhythmia": get_arrhythmia(),
        "banknote": get_banknote(),
        "bloodtransf": get_bloodtransf(),
        "breastcancer": get_breastcancer(),
        "ilpd": get_ilpd(),
        "ionosphere": get_ionosphere(),
        "parkinsons": get_parkinsons(),
        "pima": get_pima(),
        "wine": get_wine(),
    }

    return datasets
