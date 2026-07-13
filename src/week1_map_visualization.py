"""
Week 1 — Plot cleaned housing data geographically to inspect spatial pricing trends.
"""

import pandas as pd
import folium
from folium.plugins import HeatMap
from pathlib import Path

CLEAN_DATA_PATH = Path("data/processed/kc_house_clean.csv")
OUTPUT_MAP_PATH = Path("outputs/week1_price_map.html")

LAT_COL = "lat"
LONG_COL = "long"
PRICE_COL = "price"
MAX_MARKERS = 1500


def color_for_price(price, p33, p66):
    if price <= p33:
        return "green"
    elif price <= p66:
        return "orange"
    return "red"


def build_map(df: pd.DataFrame) -> folium.Map:
    center = [df[LAT_COL].mean(), df[LONG_COL].mean()]
    m = folium.Map(location=center, zoom_start=10, tiles="cartodbpositron")

    sample = df.sample(n=min(MAX_MARKERS, len(df)), random_state=42)
    p33, p66 = df[PRICE_COL].quantile([0.33, 0.66])

    marker_layer = folium.FeatureGroup(name="Price Markers (sampled)")
    for _, row in sample.iterrows():
        folium.CircleMarker(
            location=[row[LAT_COL], row[LONG_COL]],
            radius=3,
            color=color_for_price(row[PRICE_COL], p33, p66),
            fill=True,
            fill_opacity=0.6,
            popup=f"${row[PRICE_COL]:,.0f}",
        ).add_to(marker_layer)
    marker_layer.add_to(m)

    heat_data = df[[LAT_COL, LONG_COL, PRICE_COL]].values.tolist()
    heat_layer = folium.FeatureGroup(name="Price Heatmap")
    HeatMap(heat_data, radius=8, blur=10, max_zoom=13).add_to(heat_layer)
    heat_layer.add_to(m)

    folium.LayerControl().add_to(m)
    return m


def main():
    df = pd.read_csv(CLEAN_DATA_PATH)
    print(f"Loaded {len(df):,} cleaned rows for mapping")

    m = build_map(df)
    OUTPUT_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUTPUT_MAP_PATH))
    print(f"Saved interactive map -> {OUTPUT_MAP_PATH}")


if __name__ == "__main__":
    main()