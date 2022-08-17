from data_preprocessing import Dataset, train_valid_test_split
from loss import DiceLoss
from metrics import compute_dice_coefficient
from model import AttentionUNet
from training import train

from pathlib import Path

import numpy as np
import torch


IMAGES_PATH = Path("data/images/")
MASKS_PATH = Path("data/masks/")
SAVED_MODEL_PATH = Path("saved_model/saved_model.pth")

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
    model.load_state_dict(torch.load(str(SAVED_MODEL_PATH)))
    optimizer = torch.optim.Adam(model.parameters())
    compute_loss = DiceLoss()
    metrics = {"Dice Coefficient": compute_dice_coefficient}

    train(
        model,
        train_loader,
        valid_loader,
        compute_loss,
        optimizer,
        metrics,
        N_EPOCHS,
        patience=10,
        saved_model_path=SAVED_MODEL_PATH
    )


if __name__ == "__main__":
    main()
