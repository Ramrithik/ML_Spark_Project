# ML Spark Challenge — Smart Procurement: Delivery Delay Prediction
##  Given a data of 1200 ,but 1196 of them are delayed and only 4 are on time , therefore the data is highly skewed and accuracy is not a metric of evalution bcz i achieved 99.7% accuracy on svm model but it has no use 
### To over come this issue i am using Class weighting which tells the algorithm: "a mistake on the minority class is far more costly than a mistake on the majority class." It works by multiplying each sample's contribution to the loss/impurity by its class weight. 
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
