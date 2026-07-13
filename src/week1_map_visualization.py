from pathlib import Path

import folium
import pandas as pd
from folium.plugins import HeatMap, MarkerCluster


def load_cleaned_data(processed_path: Path) -> pd.DataFrame:
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {processed_path}. Run src/week1_load_clean.py first."
        )

    df = pd.read_csv(processed_path)
    print(f"Loaded {len(df):,} cleaned rows from {processed_path}")
    return df


def create_price_map(df: pd.DataFrame) -> folium.Map:
    center_lat = df["lat"].median()
    center_long = df["long"].median()
    price_min = df["price"].min()
    price_max = df["price"].max()

    m = folium.Map(location=[center_lat, center_long], zoom_start=10, tiles="CartoDB positron")

    marker_cluster = MarkerCluster(name="Price markers")
    for _, row in df.iterrows():
        price = row["price"]
        color = "green" if price <= price_min + (price_max - price_min) * 0.33 else (
            "orange" if price <= price_min + (price_max - price_min) * 0.66 else "red"
        )
        folium.CircleMarker(
            location=[row["lat"], row["long"]],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"Price: ${price:,.0f}<br>Bedrooms: {row.get('bedrooms', 'NA')}<br>Bathrooms: {row.get('bathrooms', 'NA')}",
                max_width=240,
            ),
        ).add_to(marker_cluster)
    marker_cluster.add_to(m)

    heat_data = df[["lat", "long", "price"]].values.tolist()
    HeatMap(heat_data, radius=12, blur=15, max_zoom=10).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def save_map(m: folium.Map, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"Saved map to {output_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    processed_path = repo_root / "data" / "processed" / "kc_house_data_clean.csv"
    output_path = repo_root / "outputs" / "week1_price_map.html"

    df = load_cleaned_data(processed_path)
    m = create_price_map(df)
    save_map(m, output_path)


if __name__ == "__main__":
    main()
