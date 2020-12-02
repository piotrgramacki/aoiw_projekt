import os
import shutil
from sklearn.model_selection import train_test_split
from typing import List, Tuple


def split_files(
    source_directory: str,
    destination_directory: str,
    train_percentage=0.8,
    random_seed=None,
):
    train_dest = os.path.join(destination_directory, "train")
    test_dest = os.path.join(destination_directory, "test")

    assert_directory_exists_and_empty(train_dest)
    assert_directory_exists_and_empty(test_dest)

    class_names = [
        directory
        for directory in os.listdir(source_directory)
        if os.path.isdir(os.path.join(source_directory, directory))
    ]

    X_paths_names: List[Tuple[str, str]] = []
    y_names: List[str] = []
    for class_name in class_names:
        class_source_directory = os.path.join(source_directory, class_name)
        files = [
            (os.path.join(class_source_directory, f), f)
            for f in os.listdir(class_source_directory)
            if os.path.isfile(os.path.join(class_source_directory, f))
        ]
        X_paths_names.extend(files)
        y_names.extend([class_name] * len(files))

    X_train, X_test, y_train, y_test = train_test_split(
        X_paths_names,
        y_names,
        train_size=train_percentage,
        shuffle=True,
        stratify=y_names,
        random_state=random_seed,
    )

    assert_destination_directories_correct(train_dest, test_dest, class_names)

    save_to_destination(X_train, y_train, train_dest)
    save_to_destination(X_test, y_test, test_dest)


def assert_destination_directories_correct(
    train_destination_directory: str,
    test_destination_directory: str,
    class_names: List[str],
):
    for class_name in class_names:
        assert_class_destinations_correct(
            train_destination_directory, test_destination_directory, class_name
        )


def assert_class_destinations_correct(
    train_destination_directory: str, test_destination_directory: str, class_name: str
):
    assert_destination_correct(train_destination_directory, class_name)
    assert_destination_correct(test_destination_directory, class_name)


def assert_destination_correct(destination_directory: str, class_name: str):
    directory = os.path.join(destination_directory, class_name)
    assert_directory_exists_and_empty(directory)


def assert_directory_exists_and_empty(directory: str):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)


def save_to_destination(
    X_paths_names: List[Tuple[str, str]], y_names: List[str], destination_directory: str
):
    for (x_path, x_file_name), y_class_name in zip(X_paths_names, y_names):
        file_destination_path = os.path.join(
            destination_directory, y_class_name, x_file_name
        )
        shutil.copyfile(x_path, file_destination_path)
