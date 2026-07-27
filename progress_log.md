# Progress Log

## Week 1
- Created core project structure for geospatial valuation
- Loaded and cleaned King County housing dataset (21,613 -> 21,613 rows after cleaning)
- Filtered price outliers via IQR method (1,146 rows removed, 20,467 remaining)
- Built Folium spatial price visualization map

## Week 2
- Engineered house_age, was_renovated, yrs_since_update, dist_to_center_km features
- Trained XGBoost baseline regressor
- Baseline results: MAPE = 15.30%, RMSE = $76,688.04

## Week 3
- Built a K-nearest-neighbor graph (K=8) connecting each house to its closest physical neighbors using Haversine distance
- Computed neighbor_avg_price and neighbor_price_std as spatial embedding proxies
- Visualized a 150-house sample subgraph to confirm edges reflect real geographic clustering