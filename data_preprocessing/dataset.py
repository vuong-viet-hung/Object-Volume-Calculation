from pathlib import Path
from typing import Callable, Sequence

import torch
import torchvision


IMAGE_SIZE = (128, 128)


def default_transformer(x):
    return torchvision.transforms.Resize(IMAGE_SIZE)(x) / 255


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self, image_paths: Sequence[Path],
            mask_paths: Sequence[Path],
            transformer: Callable[[torch.Tensor], torch.Tensor] = default_transformer,
    ) -> None:
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transformer = transformer

    def __len__(self) -> int:
        image_count = len(self.image_paths)
        mask_count = len(self.mask_paths)
        if image_count != mask_count:
            raise ValueError("Numbers of image and mask don't match.")
        return image_count

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = torchvision.io.read_image(str(image_path))
        mask = torchvision.io.read_image(str(mask_path))

        transformed_image = self.transformer(image)
        transformed_mask = self.transformer(mask)

        return transformed_image, transformed_mask
