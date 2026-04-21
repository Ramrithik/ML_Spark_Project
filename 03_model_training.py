import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                              f1_score, precision_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

import sys; sys.path.insert(0, "."); exec(open("02_feature_engineering.py").read().replace("if __name__", "if False"))   # reuse Step 2

os.makedirs('plots', exist_ok=True)

df, FEATURES = load_and_engineer()
X = df[FEATURES]
y = df['delay_flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

cw = dict(zip([0, 1], compute_class_weight(
    'balanced', classes=np.array([0, 1]), y=y_train)))
print(f"Class weights: {cw}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(scale_pos_weight=cw[0] / cw[1], n_estimators=200,learning_rate=0.05, max_depth=5, random_state=42,eval_metric='logloss', verbosity=0),
}

results = {}
print("\n" )
print("MODEL RESULTS")

for name, model in models.items():
    Xtr = X_train_s if 'Logistic' in name else X_train
    Xte = X_test_s  if 'Logistic' in name else X_test
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]
    results[name] = {
        'model':     model,
        'y_pred':    y_pred,
        'f1':        f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall':    recall_score(y_test, y_pred, average='weighted'),
        'roc_auc':   roc_auc_score(y_test, y_prob),
    }
    print(f"\n")
    print(f"  {name}")
    print(f"  F1={results[name]['f1']:.4f}  ROC-AUC={results[name]['roc_auc']:.4f}")
    print(classification_report(y_test, y_pred))
xgb_model = results['XGBoost']['model']
importances = pd.Series(
    xgb_model.feature_importances_, index=FEATURES).sort_values(ascending=False)

print("\nTop 10 Feature Importances (XGBoost):")
print(importances.head(10).to_string())


fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Model Performance & Feature Importance',
             fontsize=15, fontweight='bold')

ax = axes[0]
model_names = list(results.keys())
metrics = ['f1', 'roc_auc', 'precision', 'recall']
metric_labels = ['F1 (weighted)', 'ROC-AUC', 'Precision', 'Recall']
x = np.arange(len(model_names))
width = 0.2
colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']
for i, (m, label, c) in enumerate(zip(metrics, metric_labels, colors)):
    vals = [results[n][m] for n in model_names]
    ax.bar(x + i * width - 1.5 * width, vals, width, label=label, color=c,
           alpha=0.85, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels([n.replace(' ', '\n') for n in model_names])
ax.set_ylim(0.85, 1.02)
ax.set_ylabel('Score')
ax.set_title('Model Comparison')
ax.legend(loc='lower right', fontsize=8)

ax = axes[1]
top15 = importances.head(15)
bar_colors = ['#E74C3C' if v > 0.10 else '#3498DB' if v > 0.05 else '#95A5A6'
              for v in top15.values]
ax.barh(range(len(top15)), top15.values, color=bar_colors, edgecolor='white')
ax.set_yticks(range(len(top15)))
ax.set_yticklabels([n.replace('_', ' ').title() for n in top15.index], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance Score')
ax.set_title('Top 15 Features — XGBoost')
high_p = mpatches.Patch(color='#E74C3C', label='High (>10%)')
mid_p  = mpatches.Patch(color='#3498DB', label='Medium (5-10%)')
low_p  = mpatches.Patch(color='#95A5A6', label='Lower (<5%)')
ax.legend(handles=[high_p, mid_p, low_p], fontsize=8, loc='lower right')

plt.tight_layout()
plt.savefig('plots/model_results.png', bbox_inches='tight', dpi=140)
plt.close()
print("\nModel results plot saved to plots/model_results.png")

with open('model_artifacts.pkl', 'wb') as f:
    pickle.dump({
        'results': results, 'df': df, 'importances': importances,
        'X': X, 'y': y, 'X_test': X_test, 'y_test': y_test,
        'FEATURES': FEATURES, 'scaler': scaler
    }, f)
print("Model artifacts saved to model_artifacts.pkl")
