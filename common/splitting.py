import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

def split_by_case(
    csv_path: str,
    output_train_csv: str = "/home/team11/dev/MediSense/classification/temp/train_labels.csv",
    output_val_csv: str = "/home/team11/dev/MediSense/classification/temp/val_labels.csv",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Splits a dataset by case_id, copying the corresponding subfolders into
    train/ and val/ directories, and writing out separate CSV files.
    
    :param csv_path: Path to the CSV file containing labels (must have a 'case_id' column).
    :param images_dir: Path to the top-level 'images' directory, which contains subfolders named by case_id.
    :param output_train_csv: Filename for the output training CSV.
    :param output_val_csv: Filename for the output validation CSV.
    :param output_train_dir: Directory where train subfolders will be copied.
    :param output_val_dir: Directory where val subfolders will be copied.
    :param test_size: Fraction of the dataset to include in the validation split (default=0.2).
    :param random_state: Seed for reproducibility (default=42).
    """

    # 1) Load the CSV
    df = pd.read_csv(csv_path)
    if "case_id" not in df.columns:
        raise ValueError("CSV file must contain a 'case_id' column.")

    # 2) Get the unique cases and split them
    unique_cases = df["case_id"].unique()
    train_cases, val_cases = train_test_split(
        unique_cases, test_size=test_size, random_state=random_state
    )

    # 3) Create train/val DataFrames
    train_df = df[df["case_id"].isin(train_cases)]
    val_df = df[df["case_id"].isin(val_cases)]

    # 4) Remove existing CSVs if they exist
    if os.path.exists(output_train_csv):
        os.remove(output_train_csv)
    
    if os.path.exists(output_val_csv):
        os.remove(output_val_csv)

    # 5) Save them to CSV
    train_df.to_csv(output_train_csv, index=False)
    val_df.to_csv(output_val_csv, index=False)

    print(f"Created {output_train_csv} with {train_df.shape[0]} rows.")
    print(f"Created {output_val_csv} with {val_df.shape[0]} rows.")

    print("\nSplit complete. Your new train/val CSVs are ready.")

if __name__ == "__main__":
    # Example usage:
    split_by_case(
        csv_path="/home/data/train/localization.csv",
        images_dir="/home/data/train/images",
        output_train_csv="/home/team11/dev/MediSense/loc/dataset/train/localization.csv",
        output_val_csv="/home/team11/dev/MediSense/loc/dataset/subset-train-eval/localization.csv",
        test_size=0.2,
        random_state=42
    )
