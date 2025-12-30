from pathlib import Path

_UTILS_DIR = Path(__file__).parent
_SRC_DIR = _UTILS_DIR.parent
PROJECT_ROOT = _SRC_DIR.parent

DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
OUTPUT_CSV = DATA_ROOT / "kdef_split.csv"
MODELS_DIR = PROJECT_ROOT / "outputs/models"

DATA_PATHS = {
    "Original": RAW_DIR,
    "Grayscale": DATA_ROOT / "grayscale",
    "Degraded": DATA_ROOT / "degraded",
}

SPLIT_CSV = DATA_ROOT / "kdef_split.csv"
