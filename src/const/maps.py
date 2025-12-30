EMOTION_MAP = {
    "AF": "afraid",
    "AN": "angry",
    "DI": "disgusted",
    "HA": "happy",
    "NE": "neutral",
    "SA": "sad",
    "SU": "surprised",
}

EMOTION_IDX_MAP = {
    "AF": 0,
    "AN": 1,
    "DI": 2,
    "HA": 3,
    "NE": 4,
    "SA": 5,
    "SU": 6,
}

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
