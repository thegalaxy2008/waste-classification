import kagglehub
import shutil
import os
import random
import utils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import pathlib


def download_and_prepare_dataset():
    """Download and prepare the garbage classification dataset.

    """

    # Download latest version
    path = kagglehub.dataset_download("sumn2u/garbage-classification-v2")
    print("Path to dataset files:", path)

    src_folder = os.path.join(path, "garbage-dataset")
    dest_folder = os.path.join(utils.get_cwd(), "dataset", "garbage-dataset")



    # If the folder already exists, skip
    
    
    if os.path.exists(dest_folder):
        print(
            f"Dataset already exists at '{dest_folder}'. Skipping download and preparation."
        )
    else:
        os.makedirs(os.path.join(dest_folder, "train"), exist_ok=True)
        os.makedirs(os.path.join(dest_folder, "test"), exist_ok=True)

        split_ratio = 0.8  # 80% train, 20% test

        for class_name in os.listdir(src_folder):
            class_path = os.path.join(src_folder, class_name)
            print(f"Processing class '{class_name}' from '{class_path}'")
            if os.path.isdir(class_path):
                images = os.listdir(class_path)
                random.shuffle(images)
                split_idx = int(len(images) * split_ratio)
                train_images = images[:split_idx]
                test_images = images[split_idx:]

                # Create class folders in train and test
                train_class_dir = os.path.join(dest_folder, "train", class_name)
                test_class_dir = os.path.join(dest_folder, "test", class_name)
                os.makedirs(train_class_dir, exist_ok=True)
                os.makedirs(test_class_dir, exist_ok=True)

                # Copy train images
                for img in train_images:
                    shutil.copy2(
                        os.path.join(class_path, img), os.path.join(train_class_dir, img)
                    )

                # Copy test images
                for img in test_images:
                    shutil.copy2(
                        os.path.join(class_path, img), os.path.join(test_class_dir, img)
                    )


        print(f"Dataset split into train and test sets at '{dest_folder}'")
    # Download latest version
    path_1 = kagglehub.dataset_download("techsash/waste-classification-data")

    print("Path to dataset files:", path_1)

    src_folder_1 = os.path.join(path_1, "DATASET")
    dest_folder_1 = os.path.join(utils.get_cwd(), "dataset", "recycleable-and-non-recycleable-dataset")

    # If the folder already exists, skip
    if os.path.exists(dest_folder_1):
        print(
            f"Dataset already exists at '{dest_folder_1}'. Skipping download and preparation."
        )
        return
    

    shutil.copytree(src_folder_1, dest_folder_1)


def create_data_loaders(
    train_dir: pathlib.Path,
    test_dir: pathlib.Path,
    transforms: transforms.Compose,
    batch_size: int,
):
    """
    Creates data loaders for training and testing from image directories.

    Args:
        train_dir (pathlib.Path): Path to training data directory
        test_dir (pathlib.Path): Path to test data directory
        transforms (torchvision.transforms.Compose): Transforms to apply to the images
        batch_size (int): Batch size for data loaders

    Returns:
        tuple[DataLoader, DataLoader, list[str]]: Train loader, test loader, and class names
    """
    # Create training dataset
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transforms)

    # Create test dataset
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transforms)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_dataset.classes


def create_partial_data_loaders(
    train_dir: pathlib.Path,
    test_dir: pathlib.Path,
    transforms: transforms.Compose,
    batch_size: int,
    ratio: float = 0.1,
):
    """
    Creates data loaders for training and testing from image directories, using only a portion of the dataset.
    Args:
        train_dir (pathlib.Path): Path to training data directory
        test_dir (pathlib.Path): Path to test data directory
        transforms (torchvision.transforms.Compose): Transforms to apply to the images
        batch_size (int): Batch size for data loaders
        ratio (float): Ratio of the dataset to use (default: 0.1)
    Returns:
        tuple[DataLoader, DataLoader, list[str]]: Train loader, test loader, and class names
    """
    # Create full training dataset
    full_train_dataset = datasets.ImageFolder(root=train_dir, transform=transforms)
    full_test_dataset = datasets.ImageFolder(root=test_dir, transform=transforms)

    # Create a subset of the training dataset
    partial_train_dataset = torch.utils.data.random_split(
        full_train_dataset, [ratio, 1 - ratio]
    )[0]

    # Create a subset of the test dataset
    partial_test_dataset = torch.utils.data.random_split(
        full_test_dataset, [ratio, 1 - ratio]
    )[0]

    # Create data loaders
    train_loader = DataLoader(
        partial_train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(partial_test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, full_train_dataset.classes


if __name__ == "__main__":
    download_and_prepare_dataset()
#cross-import code to not run this 