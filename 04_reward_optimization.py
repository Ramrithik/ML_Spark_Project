import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import os

os.makedirs('plots', exist_ok=True)
with open('model_artifacts.pkl', 'rb') as f:
    data = pickle.load(f)

results     = data['results']
df          = data['df']
importances = data['importances']
X_test      = data['X_test']
y_test      = data['y_test']

xgb_model = results['XGBoost']['model']

df_test = df.iloc[X_test.index].copy()
df_test['delay_prob']      = xgb_model.predict_proba(X_test)[:, 1]
df_test['predicted_delay'] = xgb_model.predict(X_test)


def compute_reward(row):
    reward = 10 if row['delay_flag'] == 0 else -15
    if row['prj_priority_level'] == 'High':
        reward += 5
    excess = max(0, row['actual_time_hours'] - row['expected_time_hours'])
    reward -= 2 * int(excess)
    return reward


df_test['reward'] = df_test.apply(compute_reward, axis=1)

print("REWARD ANALYSIS")

print(f"Average reward : {df_test['reward'].mean():.2f}")
print(f"Min reward     : {df_test['reward'].min():.0f}")
print(f"Max reward     : {df_test['reward'].max():.0f}")
print(f"\nReward by priority:")
print(df_test.groupby('prj_priority_level')['reward'].mean().to_string())

df_test['priority_score'] = (
    df_test['priority_encoded'] * 3 +                                             
    (1 - df_test['delay_prob']) * 5 +                                             
    (1 / (df_test['distance_km'] / df_test['distance_km'].max() + 0.01)) * 2      
)

print("\n" )
print("TOP 10 DELIVERIES TO DISPATCH FIRST")
top10 = df_test.nlargest(10, 'priority_score')[
    ['delivery_id', 'factory_id', 'project_id', 'distance_km',
     'prj_priority_level', 'delay_prob', 'priority_score']
].reset_index(drop=True)
print(top10.to_string(index=False))

print("\n--- Dispatch Strategy ---")
print("HIGH priority + LOW delay_prob  -> Dispatch immediately")
print("HIGH priority + HIGH delay_prob -> Pre-position from nearest factory")
print("LOW  priority + HIGH delay_prob -> Reschedule or reroute")


fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Reward-Based Delivery Optimization', fontsize=15, fontweight='bold')

ax = axes[0]
ax.hist(df_test['reward'], bins=25, color='#3498DB', edgecolor='white', alpha=0.85)
ax.axvline(df_test['reward'].mean(), color='#E74C3C', linestyle='--', linewidth=2,
           label=f'Mean: {df_test["reward"].mean():.1f}')
ax.set_title('Reward Score Distribution (Test Set)')
ax.set_xlabel('Reward Score')
ax.set_ylabel('Count')
ax.legend()


ax = axes[1]
reward_prio = df_test.groupby('prj_priority_level')['reward'].mean()
for p, c in zip(['Low', 'Medium', 'High'], ['#95A5A6', '#F39C12', '#E74C3C']):
    if p in reward_prio.index:
        ax.bar(p, reward_prio[p], color=c, edgecolor='white', width=0.5)
        ax.text(p, reward_prio[p] + 0.3, f'{reward_prio[p]:.1f}',
                ha='center', fontweight='bold')
ax.set_title('Average Reward by Priority Level')
ax.set_ylabel('Avg Reward Score')
ax.axhline(0, color='black', linewidth=0.8, linestyle=':')

ax = axes[2]
top15 = df_test.nlargest(15, 'priority_score')
bar_colors = ['#E74C3C' if p == 'High' else '#F39C12' if p == 'Medium' else '#3498DB'
              for p in top15['prj_priority_level']]
ax.barh(range(len(top15)), top15['priority_score'].values,
        color=bar_colors, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15['delivery_id'].values, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Priority Score')
ax.set_title('Top 15 Deliveries to Prioritize')
high_p = mpatches.Patch(color='#E74C3C', label='High priority')
med_p  = mpatches.Patch(color='#F39C12', label='Medium priority')
low_p  = mpatches.Patch(color='#3498DB', label='Low priority')
ax.legend(handles=[high_p, med_p, low_p], fontsize=8)

plt.tight_layout()
plt.savefig('plots/reward_optimization.png', bbox_inches='tight', dpi=140)
plt.close()
print("\nReward plot saved to plots/reward_optimization.png")
