import pandas as pd
import torch
from deepface import DeepFace
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from src.utils.constants import (
    DATA_PATHS,
    DEEPFACE_MAP,
    EMOTION_IDX_MAP,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

transform_eval = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def evaluate(
    df_test: pd.DataFrame,
    device: torch.device,
    model_frozen: torch.nn.Module,
    model_unfrozen: torch.nn.Module,
) -> pd.DataFrame:
    results = []

    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        rel_path = row["relative_path"]
        true_label_str = row["emotion_code"]
        true_label_idx = EMOTION_IDX_MAP[true_label_str]

        for domain_name, root_dir in DATA_PATHS.items():
            img_path = root_dir / rel_path

            if not img_path.exists():
                continue

            img_pil = Image.open(img_path).convert("RGB")
            img_tensor = transform_eval(img_pil).unsqueeze(0).to(device)  # type: ignore

            with torch.no_grad():
                logits_frozen = model_frozen(img_tensor)
                pred_frozen_idx = torch.argmax(logits_frozen, dim=1).item()

                logits_unfrozen = model_unfrozen(img_tensor)
                pred_unfrozen_idx = torch.argmax(logits_unfrozen, dim=1).item()

            try:
                dfs = DeepFace.analyze(
                    img_path=str(img_path),
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv",
                    silent=True,
                )
                deepface_emotion = dfs[0]["dominant_emotion"]  # type: ignore

                deepface_code = DEEPFACE_MAP.get(deepface_emotion, "UNK")
                pred_deepface_idx = EMOTION_IDX_MAP.get(deepface_code, -1)

            except Exception:
                pred_deepface_idx = -1

            results.append(
                {
                    "file": str(rel_path),
                    "subject": row["subject_id"],
                    "domain": domain_name,
                    "true_idx": true_label_idx,
                    "true_str": true_label_str,
                    "pred_frozen": pred_frozen_idx,
                    "pred_unfrozen": pred_unfrozen_idx,
                    "pred_deepface": pred_deepface_idx,
                }
            )

    return pd.DataFrame(results)
