import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.const.maps import BUGGED_FILES, TARGET_ANGLES
from src.const.paths import OUTPUT_CSV, RAW_DIR


def _read_data() -> pd.DataFrame:
    data = []
    for p in RAW_DIR.rglob("*.JPG"):
        name = p.stem

        if name in BUGGED_FILES:
            print("Skipped buggy file:", p)
            continue

        rel_path = p.relative_to(RAW_DIR)
        angle = name[6:]

        if angle in TARGET_ANGLES:
            data.append(
                {
                    "relative_path": str(rel_path),
                    "subject_id": name[1:4],
                    "emotion_code": name[4:6],
                }
            )

    return pd.DataFrame(data)


def generate_split(seed: int = 42) -> None:
    """
    Generate train/val/test split ensuring no subject overlap.

    Args:
        seed (int): Random seed for reproducibility.

    """
    df = _read_data()

    print(f"Total images found: {len(df)}")
    print(f"Unique subjects: {df['subject_id'].nunique()}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, temp_idx = next(splitter.split(df, groups=df["subject_id"]))

    df.loc[train_idx, "split"] = "train"
    temp_df = df.iloc[temp_idx]

    splitter_val = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_sub_idx, test_sub_idx = next(
        splitter_val.split(temp_df, groups=temp_df["subject_id"])
    )

    val_subjects = temp_df.iloc[val_sub_idx]["subject_id"].unique()
    test_subjects = temp_df.iloc[test_sub_idx]["subject_id"].unique()

    df.loc[df["subject_id"].isin(val_subjects), "split"] = "val"
    df.loc[df["subject_id"].isin(test_subjects), "split"] = "test"

    print("\nSplit distribution:")
    print(df["split"].value_counts())

    train_subs = set(df[df["split"] == "train"]["subject_id"])
    test_subs = set(df[df["split"] == "test"]["subject_id"])
    assert len(train_subs.intersection(test_subs)) == 0, "DATA LEAKAGE DETECTED!"

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved master split to {OUTPUT_CSV}")
