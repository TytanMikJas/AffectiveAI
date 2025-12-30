import os
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.const.maps import EMOTION_IDX_MAP
from src.const.transforms import TRAIN_TRANSFORM, VAL_TRANSFORM


class KDEFDataset(Dataset):
    """
    PyTorch Dataset for the KDEF (Karolinska Directed Emotional Faces) dataset.
    """

    def __init__(self, df: pd.DataFrame, transform: transforms.Compose):
        """
        Initialize the KDEF Dataset.

        Args:
            df (pd.DataFrame): DataFrame containing a 'path' column with file paths to images.
            transform (transforms.Compose): transform to be applied on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
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

        return image, label_tensor  # type: ignore


class KDEFDataModule(pl.LightningDataModule):
    """
    DataModule to manage data transformations and DataLoader creation for KDEF dataset.
    """

    def __init__(
        self,
        csv_file: str = "data/kdef_split.csv",
        root_dir: str = "data/raw",
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """
        Initialize the DataModule.

        Args:
            csv_file (str): Path to the CSV file containing data splits.
            root_dir (str): Root directory where images are stored.
            batch_size (int): Size of batches for the DataLoaders.
            num_workers (int): Number of subprocesses to use for data loading.
        """
        super().__init__()

        self.csv_file = csv_file
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        full_df = pd.read_csv(self.csv_file)

        full_df["path"] = full_df["relative_path"].apply(lambda x: self.root_dir / x)

        self.train_df = full_df[full_df["split"] == "train"]
        self.val_df = full_df[full_df["split"] == "val"]
        self.test_df = full_df[full_df["split"] == "test"]

        print(f"Loaded data from {self.root_dir}:")
        print(
            f"Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}"
        )

    def train_dataloader(self):
        return DataLoader(
            KDEFDataset(self.train_df, transform=TRAIN_TRANSFORM),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            KDEFDataset(self.val_df, transform=VAL_TRANSFORM),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            KDEFDataset(self.test_df, transform=VAL_TRANSFORM),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )
