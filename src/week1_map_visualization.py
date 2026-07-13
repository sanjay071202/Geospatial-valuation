"""
Week 1: Map Visualization
Create a Folium map visualization for geospatial valuation data.
"""

import pandas as pd
import folium
from pathlib import Path


def load_cleaned_data(filepath):
    """
    Load the cleaned house data.
    
    Args:
        filepath (str): Path to the cleaned CSV file
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None


def create_map(df, center_lat=47.6062, center_lon=-122.3321, zoom_start=10):
    """
    Create a Folium map with house price data.
    
    Args:
        df (pd.DataFrame): House data with latitude, longitude, and price columns
        center_lat (float): Center latitude for the map
        center_lon (float): Center longitude for the map
        zoom_start (int): Initial zoom level
        
    Returns:
        folium.Map: Folium map object
    """
    # Create base map (centered on Seattle)
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap"
    )
    
    # TODO: Add markers or heatmap layer
    # Example:
    # for idx, row in df.iterrows():
    #     folium.CircleMarker(
    #         location=[row['lat'], row['lon']],
    #         radius=5,
    #         popup=f"Price: ${row['price']:,.0f}",
    #         color='blue',
    #         fill=True,
    #         fillOpacity=0.7
    #     ).add_to(m)
    
    return m


def save_map(map_obj, output_path):
    """
    Save the Folium map to HTML file.
    
    Args:
        map_obj (folium.Map): Folium map object
        output_path (str): Path to save the HTML file
    """
    map_obj.save(output_path)
    print(f"Map saved to {output_path}")


if __name__ == "__main__":
    # Define paths
    cleaned_data_path = Path(__file__).parent.parent / "data" / "processed" / "cleaned_data.csv"
    output_map_path = Path(__file__).parent.parent / "outputs" / "week1_price_map.html"
    
    # Load cleaned data
    df = load_cleaned_data(cleaned_data_path)
    
    if df is not None:
        # Create map
        map_obj = create_map(df)
        
        # Save map
        save_map(map_obj, output_map_path)
