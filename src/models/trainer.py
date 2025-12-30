import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Precision, Recall
from torchvision import models


class EmotionClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 7,
        learning_rate: float = 1e-3,
        freeze_backbone: bool = True,
    ):
        """
        MobileNetV3 Large wrapper for Emotion Recognition.

        Args:
            num_classes (int): Number of emotion categories.
            learning_rate (float): Learning rate for Adam optimizer.
            freeze_backbone (bool): If True, freezes the feature extractor.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        )
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, num_classes)  # type: ignore

        metrics_kwargs = {
            "task": "multiclass",
            "num_classes": num_classes,
            "average": "macro",
        }
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(**metrics_kwargs)
        self.val_precision = Precision(**metrics_kwargs)
        self.val_recall = Recall(**metrics_kwargs)

        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(**metrics_kwargs)

        self.conf_mat = ConfusionMatrix(task="multiclass", num_classes=num_classes)

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freezes feature extractor parameters."""
        for param in self.model.features.parameters():
            param.requires_grad = False
        print("INFO: Backbone frozen. Training classifier head only.")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)

        self.train_acc(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)

        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.val_precision(logits, y)
        self.val_recall(logits, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.test_acc(logits, y)
        self.test_f1(logits, y)

        self.conf_mat.update(logits, y)

        self.log("test_acc", self.test_acc)
        self.log("test_f1", self.test_f1)

    def configure_optimizers(self):  # type: ignore
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.Adam(params, lr=self.hparams.learning_rate)  # type: ignore

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def train(self, mode: bool = True):  # type: ignore
        super().train(mode)

        if self.hparams.freeze_backbone:  # type: ignore
            for module in self.model.features.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
