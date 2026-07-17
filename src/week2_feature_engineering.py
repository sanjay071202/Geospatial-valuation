"""
Week 2 — Engineer tabular features for the baseline model.
"""

import pandas as pd
import numpy as np
from pathlib import Path

CLEAN_DATA_PATH = Path("data/processed/kc_house_clean.csv")
FEATURED_DATA_PATH = Path("data/processed/kc_house_features.csv")

# Downtown Seattle coordinates — reference point for distance-to-center feature
CITY_CENTER_LAT = 47.6062
CITY_CENTER_LONG = -122.3321


def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometers between two lat/long points."""
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) * 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) * 2
    return 2 * R * np.arcsin(np.sqrt(a))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # House age at time of sale
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sale_year"] = df["date"].dt.year
    df["house_age"] = df["sale_year"] - df["yr_built"]

    # Renovation flag
    df["was_renovated"] = (df["yr_renovated"] > 0).astype(int)

    # Years since last renovation (or since built, if never renovated)
    df["yrs_since_update"] = df["sale_year"] - df[["yr_built", "yr_renovated"]].max(axis=1)

    # Distance to city center (km) — reused later for KNN graph logic in Week 3
    df["dist_to_center_km"] = haversine_distance(
        df["lat"], df["long"], CITY_CENTER_LAT, CITY_CENTER_LONG
    )

    return df


def main():
    df = pd.read_csv(CLEAN_DATA_PATH)
    print(f"Loaded {len(df):,} cleaned rows")

    df = engineer_features(df)
    print("Engineered features: house_age, was_renovated, yrs_since_update, dist_to_center_km")

    FEATURED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURED_DATA_PATH, index=False)
    print(f"Saved featured dataset -> {FEATURED_DATA_PATH} ({len(df):,} rows)")


if __name__ == "__main__":
    main()