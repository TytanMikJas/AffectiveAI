from pathlib import Path

EMOTION_MAP = {
    "AF": "afraid",
    "AN": "angry",
    "DI": "disgusted",
    "HA": "happy",
    "NE": "neutral",
    "SA": "sad",
    "SU": "surprised",
}
EMOTION_IDX_MAP = {"AF": 0, "AN": 1, "DI": 2, "HA": 3, "NE": 4, "SA": 5, "SU": 6}
IDX_TO_EMOTION = {v: k for k, v in EMOTION_IDX_MAP.items()}
DEEPFACE_MAP = {
    "fear": "AF",
    "angry": "AN",
    "disgust": "DI",
    "happy": "HA",
    "neutral": "NE",
    "sad": "SA",
    "surprise": "SU",
}

TARGET_ANGLES = ["S", "HL", "HR"]
ANGLE_MAP = {
    "FL": "full left profile",
    "HL": "half left profile",
    "S.": "straight profile",
    "HR": "half right profile",
    "FR": "full right profile",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BUGGED_FILES = [
    "AF01SUFR",
    "AF10AFFR",
    "AF11NEHL",
    "AF20DIHL",
    "AM25DIFL",
    "AM34DIFR",
    "BF13NEHR",
    "BM21DIFL",
    "BM22DIHL",
    "BM24DIFL",
]

SPLIT_CSV = Path("data/kdef_split.csv")
MODELS_DIR = Path("outputs/models")
DATA_PATHS = {
    "Original": Path("data/raw"),
    "Grayscale": Path("data/grayscale"),
    "Degraded": Path("data/degraded"),
}
CKPT_FROZEN = MODELS_DIR / "mobilenet_v3_kdef-frozen-epoch=19-val_f1=0.62.ckpt"
CKPT_UNFROZEN = MODELS_DIR / "mobilenet_v3_kdef-unfrozen-epoch=41-val_f1=0.87.ckpt"
