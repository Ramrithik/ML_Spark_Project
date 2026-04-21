# ML Spark Challenge — Smart Procurement: Delivery Delay Prediction

## Project Structure
```
ML_Spark_Project/
├── README.md
├── run_all.py                  # Run entire pipeline in one command
├── 01_eda.py                   # Exploratory Data Analysis
├── 02_feature_engineering.py   # Feature engineering (importable module)
├── 03_model_training.py        # Model training & evaluation
├── 04_reward_optimization.py   # Reward-based dispatch optimization
├── Deliveries.csv              # 1,200 delivery records (main dataset)
├── Factories.csv               # 5 factory records
├── Projects.csv                # 200 project sites
├── External_Factors.csv        # 30 days of weather & traffic indices
└── plots/                      # Auto-generated visualizations
    ├── eda_dashboard.png
    ├── model_results.png
    └── reward_optimization.png
```

## Setup
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

## Run
```bash
python run_all.py
```

Or run individual steps:
```bash
python 01_eda.py
python 02_feature_engineering.py
python 03_model_training.py
python 04_reward_optimization.py
```

## Results
- **Best Model**: XGBoost (ROC-AUC: 0.996, F1: 0.994)
- **Top Delay Drivers**: Distance × Traffic, Traffic Index, Distance
- **Outputs**: Trained model saved as `model_artifacts.pkl`, plots in `plots/`

## Reward Framework
| Event | Score |
|---|---|
| On-time delivery | +10 |
| Delayed delivery | -15 |
| High-priority project served | +5 |
| Each excess hour | -2 |
