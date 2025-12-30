from pathlib import Path

import pandas as pd
import torch
from deepface import DeepFace
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from src.const.maps import DEEPFACE_MAP, EMOTION_IDX_MAP
from src.const.paths import DATA_PATHS
from src.const.transforms import VAL_TRANSFORM


class MultiDomainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: Path, transform: transforms.Compose):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row["relative_path"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, row["relative_path"]


def _run_pytorch_inference(
    df: pd.DataFrame,
    models_dict: dict,
    domain_name: str,
    root_dir: Path,
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    """Run inference using PyTorch models in batches.

    Args:
        df (pd.DataFrame): DataFrame containing the data to be processed.
        models_dict (dict): Dictionary of PyTorch models to use for inference.
        domain_name (str): Name of the domain being processed.
        root_dir (Path): Root directory containing the images.
        device (torch.device): Device to run the models on.
        batch_size (int, optional): Batch size for inference. Defaults to 32.

    Returns:
        dict: Dictionary containing the inference results for each model.
    """
    dataset = MultiDomainDataset(df, root_dir, VAL_TRANSFORM)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    results = {}

    for m in models_dict.values():
        m.eval()
        m.to(device)

    with torch.no_grad():
        for batch_imgs, batch_paths in tqdm(loader, desc=f"PyTorch {domain_name}"):
            batch_imgs = batch_imgs.to(device)

            for model_name, model in models_dict.items():
                logits = model(batch_imgs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                if model_name not in results:
                    results[model_name] = []
                results[model_name].extend(preds)

    return results


def evaluate_models(
    df_test: pd.DataFrame,
    device: torch.device,
    model_frozen: torch.nn.Module,
    model_unfrozen: torch.nn.Module,
) -> pd.DataFrame:
    """
    Evaluate multiple models (frozen, unfrozen, DeepFace) across different data domains.

    Args:
        df_test (pd.DataFrame): DataFrame containing test data information.
        device (torch.device): Device to run the models on.
        model_frozen (torch.nn.Module): PyTorch model with frozen backbone.
        model_unfrozen (torch.nn.Module): PyTorch model with unfrozen backbone.

    Returns:
        pd.DataFrame: DataFrame containing true labels and predictions from all models.

    """

    final_rows = []

    pt_models = {"frozen": model_frozen, "unfrozen": model_unfrozen}

    for domain_name, root_dir in DATA_PATHS.items():
        if not root_dir.exists():
            continue

        pt_results = _run_pytorch_inference(
            df_test, pt_models, domain_name, root_dir, device
        )

        for i, (idx, row) in tqdm(
            enumerate(df_test.iterrows()),
            total=len(df_test),
            desc=f"DeepFace {domain_name}",
        ):
            rel_path = row["relative_path"]
            img_path = root_dir / rel_path

            pred_frozen = pt_results["frozen"][i]
            pred_unfrozen = pt_results["unfrozen"][i]

            pred_deepface = -1
            try:
                dfs = DeepFace.analyze(
                    img_path=str(img_path),
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv",
                    silent=True,
                )
                dom_emotion = dfs[0]["dominant_emotion"]  # type: ignore

                code = DEEPFACE_MAP.get(dom_emotion, "UNK")
                pred_deepface = EMOTION_IDX_MAP.get(code, -1)

            except Exception:
                pass  # -1

            final_rows.append(
                {
                    "file": rel_path,
                    "subject": row["subject_id"],
                    "domain": domain_name,
                    "true_idx": EMOTION_IDX_MAP[row["emotion_code"]],
                    "pred_frozen": pred_frozen,
                    "pred_unfrozen": pred_unfrozen,
                    "pred_deepface": pred_deepface,
                }
            )

    return pd.DataFrame(final_rows)
