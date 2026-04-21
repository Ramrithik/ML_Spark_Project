import pandas as pd
import numpy as np

def load_and_engineer():
    df   = pd.read_csv('Deliveries.csv')
    proj = pd.read_csv('Projects.csv')
    fact = pd.read_csv('Factories.csv')
    ext  = pd.read_csv('External_Factors.csv')
    df = df.merge(
        fact.add_prefix('fac_').rename(columns={'fac_factory_id': 'factory_id'}),
        on='factory_id')
    df = df.merge(
        proj.add_prefix('prj_').rename(columns={'prj_project_id': 'project_id'}),
        on='project_id')
    df = df.merge(ext, on='date')
    priority_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['priority_encoded'] = df['prj_priority_level'].map(priority_map)
    df['expected_speed_kmh']    = df['distance_km'] / df['expected_time_hours']
    df['weather_x_traffic']     = df['weather_index'] * df['traffic_index']
    df['dist_x_traffic']        = df['distance_km'] * df['traffic_index']
    df['dist_x_weather']        = df['distance_km'] * df['weather_index']
    df['prod_variability_risk'] = df['fac_production_variability'] * df['distance_km']
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week']   = df['date'].dt.dayofweek
    df['week_of_month'] = df['date'].dt.day // 7

    FEATURES = [
        'distance_km', 'expected_time_hours',
        'fac_base_production_per_week', 'fac_production_variability', 'fac_max_storage',
        'fac_latitude', 'fac_longitude',
        'prj_demand', 'priority_encoded',
        'prj_latitude', 'prj_longitude',
        'weather_index', 'traffic_index',
        'expected_speed_kmh', 'weather_x_traffic',
        'dist_x_traffic', 'dist_x_weather',
        'prod_variability_risk', 'day_of_week', 'week_of_month'
    ]
    return df, FEATURES

if __name__ == '__main__':
    df, FEATURES = load_and_engineer()
    X = df[FEATURES]
    y = df['delay_flag']
    print("FEATURE ENGINEERING COMPLETE")
    print(f"Feature matrix shape : {X.shape}")
    print(f"Target distribution  : {y.value_counts().to_dict()}")
    print(f"\nFeatures ({len(FEATURES)}):")
    for i, f in enumerate(FEATURES, 1):
        print(f"  {i:2d}. {f}")

    print("\nSample (first 3 rows):")
    print(X.head(3).to_string())
