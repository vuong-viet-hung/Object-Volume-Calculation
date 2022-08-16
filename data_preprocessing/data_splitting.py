import numpy as np
from sklearn.model_selection import train_test_split


def train_valid_test_split(
    data_paths: np.ndarray,
    labels: np.ndarray,
    valid_size: float,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the dataset into train, valid and test sets"""
    # Split the dataset into train + valid and test sets
    (
        train_valid_data_paths,
        test_data_paths,
        train_valid_labels,
        test_labels,
    ) = train_test_split(
        data_paths,
        labels,
        test_size=test_size,
    )

    # Split the train + valid set into train and valid sets
    train_data_paths, valid_data_paths, train_labels, valid_labels = train_test_split(
        train_valid_data_paths,
        train_valid_labels,
        test_size=valid_size / (1 - test_size),
    )

    return (
        train_data_paths,
        valid_data_paths,
        test_data_paths,
        train_labels,
        valid_labels,
        test_labels,
    )
