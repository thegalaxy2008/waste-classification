import kagglehub
import shutil
import os
import random
import utils
from torch.utils.data import DataLoader
from torchvision import datasets



def download_and_prepare_dataset():
    # Download latest version
    path = kagglehub.dataset_download("sumn2u/garbage-classification-v2")
    print("Path to dataset files:", path)

    src_folder = os.path.join(path, "garbage-dataset")
    dest_folder = os.path.join(utils.get_cwd(), "dataset", "garbage-dataset")

    # If the folder already exists, skip
    if os.path.exists(dest_folder):
        print(f"Dataset already exists at '{dest_folder}'. Skipping download and preparation.")
        return
    
    

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
                shutil.copy2(os.path.join(class_path, img), os.path.join(train_class_dir, img))

            # Copy test images
            for img in test_images:
                shutil.copy2(os.path.join(class_path, img), os.path.join(test_class_dir, img))

    print(f"Dataset split into train and test sets at '{dest_folder}'")
def create_data_loaders(train_dir, test_dir, transforms, batch_size):
        
    # Create training dataset
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=transforms
    )
    
    # Create test dataset
    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=transforms
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,    
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader



if __name__ == "__main__":
    download_and_prepare_dataset()


