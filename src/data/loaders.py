import os

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils.constants import (
    EMOTION_IDX_MAP,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


class KDEFDataset(Dataset):
    """
    PyTorch Dataset for the KDEF (Karolinska Directed Emotional Faces) dataset.
    """

    def __init__(self, df: pd.DataFrame, transform: transforms.Compose):
        """
        Initialize the KDEF Dataset.

        Args:
            df (pd.DataFrame): DataFrame containing a 'path' column with file paths to images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, idx: int
    ) -> tuple[Tensor, Tensor]:  # Zmieniłem type hint na Tensor, Tensor
        img_path = str(self.df.iloc[idx]["path"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Error loading image at {img_path}: {e}")

        filename = os.path.basename(img_path)
        emotion_code = filename[4:6]

        label = EMOTION_IDX_MAP.get(emotion_code)
        if label is None:
            raise ValueError(
                f"Unknown emotion code '{emotion_code}' in file {filename}"
            )

        if self.transform:
            image = self.transform(image)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return image, label_tensor # type: ignore


class KDEFDataModule(pl.LightningDataModule):
    """
    DataModule to manage data transformations and DataLoader creation for KDEF dataset.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """
        Initialize the DataModule.

        Args:
            train_df (pd.DataFrame): Training data.
            val_df (pd.DataFrame): Validation data.
            test_df (pd.DataFrame): Test data.
            batch_size (int): Size of batches for the DataLoaders.
            num_workers (int): Number of subprocesses to use for data loading.
        """
        super().__init__()

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _get_transforms(self) -> dict[str, transforms.Compose]:
        """
        Define the data transformation pipelines for train, val, and test sets.

        Returns:
            dict[str, transforms.Compose]: Dictionary containing transformations.
        """

        return {
            "train": transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            ),
        }

    def train_dataloader(self):
        return DataLoader(
            KDEFDataset(self.train_df, transform=self._get_transforms()["train"]),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            KDEFDataset(self.val_df, transform=self._get_transforms()["val"]),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            KDEFDataset(self.test_df, transform=self._get_transforms()["test"]),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )
