from pathlib import Path

import numpy as np
import pandas as pd


def load_data(raw_path: Path) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. Place kc_house_data.csv in data/raw/"
        )

    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df):,} rows from {raw_path}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # Core cleaning
    df = df.drop_duplicates()
    df = df.dropna(subset=["price", "sqft_living", "lat", "long"])
    df = df[df["price"] > 0]
    df = df[df["sqft_living"] > 0]

    # Filter by plausible King County coordinate range
    df = df[(df["lat"] >= 45.0) & (df["lat"] <= 49.0)]
    df = df[(df["long"] >= -124.0) & (df["long"] <= -120.0)]

    after_clean = len(df)
    print(f"Dropped {before - after_clean:,} rows during initial cleaning")

    price = df["price"]
    q1 = price.quantile(0.25)
    q3 = price.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df = df[(price >= lower_bound) & (price <= upper_bound)]
    after_iqr = len(df)
    print(
        f"Removed {after_clean - after_iqr:,} price outliers using IQR bounds"
        f" ({lower_bound:,.0f} to {upper_bound:,.0f})"
    )

    df = df.reset_index(drop=True)
    return df


def save_cleaned_data(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset to {output_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_path = repo_root / "data" / "raw" / "kc_house_data.csv"
    output_path = repo_root / "data" / "processed" / "kc_house_data_clean.csv"

    df = load_data(raw_path)
    cleaned = clean_data(df)
    save_cleaned_data(cleaned, output_path)
    print(f"Final cleaned dataset contains {len(cleaned):,} rows")


if __name__ == "__main__":
    main()
