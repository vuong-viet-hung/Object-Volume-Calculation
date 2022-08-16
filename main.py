from data_preprocessing import Dataset, train_valid_test_split
from loss import DiceLoss
from model import AttentionUNet
from training import train

from pathlib import Path

import numpy as np
import torch


IMAGES_PATH = Path("data/images/")
MASKS_PATH = Path("data/masks/")

# Hyperparameters
BATCH_SIZE = 8
N_EPOCHS = 100


def main() -> None:
    # Prepare the data
    image_paths = list(IMAGES_PATH.glob("*.jpg"))
    mask_paths = []

    for image_path in image_paths:
        sample_name = image_path.name
        mask_path = MASKS_PATH / sample_name
        if not mask_path.exists():
            raise ValueError(f"Image {sample_name!r} doesn't have a corresponding mask.")
        mask_paths.append(mask_path)

    (
        train_image_paths,
        valid_image_paths,
        test_image_paths,
        train_mask_paths,
        valid_mask_paths,
        test_mask_paths
    ) = train_valid_test_split(np.array(image_paths), np.array(mask_paths), valid_size=0.1, test_size=0.1)

    train_dataset = Dataset(train_image_paths, train_mask_paths)
    valid_dataset = Dataset(valid_image_paths, valid_mask_paths)
    test_dataset = Dataset(test_image_paths, test_mask_paths)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Train the model
    print(torch.cuda.get_device_name(0))
    model = AttentionUNet()
    optimizer = torch.optim.Adam(model.parameters())
    compute_loss = DiceLoss()
    metrics = {}

    train(model, train_loader, valid_loader, compute_loss, optimizer, metrics, N_EPOCHS)


if __name__ == "__main__":
    main()
