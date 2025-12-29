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
