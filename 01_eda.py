import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
sns.set_theme(style='whitegrid')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.titlesize': 13,'axes.titleweight': 'bold', 'figure.dpi': 120})
os.makedirs('plots', exist_ok=True)
df   = pd.read_csv('Deliveries.csv')
proj = pd.read_csv('Projects.csv')
fact = pd.read_csv('Factories.csv')
ext  = pd.read_csv('External_Factors.csv')
print("DATASET OVERVIEW")
print(f"Deliveries : {df.shape}")
print(f"Projects   : {proj.shape}")
print(f"Factories  : {fact.shape}")
print(f"Ext Factors: {ext.shape}")
print("\n--- Target Variable ---")
print(df['delay_flag'].value_counts())
print(f"Delay rate: {df['delay_flag'].mean()*100:.1f}%")
df['delay_ratio'] = df['actual_time_hours'] / df['expected_time_hours']
print("\n--- Delay Ratio (actual / expected) ---")
print(df[df['delay_flag']==1]['delay_ratio'].describe())
print("\n--- On-time rows ---")
print(df[df['delay_flag']==0][['distance_km','expected_time_hours','actual_time_hours','delay_ratio']])
print("\n--- Factory Stats ---")
print(fact.to_string(index=False))
print("\n--- Priority Distribution ---")
print(proj['priority_level'].value_counts())
palette = {'delayed': '#E74C3C', 'ontime': '#2ECC71','accent': '#3498DB', 'dark': '#2C3E50'}
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Exploratory Data Analysis — Delivery Dataset',
             fontsize=16, fontweight='bold', y=1.01)
ax = axes[0, 0]
fact_delay = df.groupby('factory_id')['delay_flag'].mean().reset_index()
bars = ax.bar(fact_delay['factory_id'], fact_delay['delay_flag'] * 100,
              color=palette['delayed'], edgecolor='white', linewidth=0.8, width=0.6)
ax.set_title('Delay Rate by Factory')
ax.set_ylabel('Delay Rate (%)')
ax.set_ylim(90, 102)
for bar, val in zip(bars, fact_delay['delay_flag'] * 100):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax = axes[0, 1]
prio_delay = df.merge(proj[['project_id', 'priority_level']], on='project_id')
prio_agg = prio_delay.groupby('priority_level')['delay_flag'].mean()
for p, c in zip(['Low', 'Medium', 'High'], ['#3498DB', '#F39C12', '#E74C3C']):
    bar = ax.bar(p, prio_agg.get(p, 0) * 100, color=c, edgecolor='white', width=0.5)
    ax.text(p, prio_agg.get(p, 0) * 100 + 0.1,
            f'{prio_agg.get(p,0)*100:.1f}%', ha='center', va='bottom', fontsize=9)
ax.set_title('Delay Rate by Project Priority')
ax.set_ylabel('Delay Rate (%)')
ax.set_ylim(90, 102)
ax = axes[0, 2]
ax.hist(df[df['delay_flag']==1]['distance_km'], bins=30, alpha=0.7,
        color=palette['delayed'], label='Delayed', density=True)
ax.hist(df[df['delay_flag']==0]['distance_km'], bins=5, alpha=0.9,
        color=palette['ontime'], label='On-time', density=True)
ax.set_title('Distance Distribution by Outcome')
ax.set_xlabel('Distance (km)')
ax.set_ylabel('Density')
ax.legend()
ax = axes[1, 0]
ext_s = ext.copy()
ext_s['date'] = pd.to_datetime(ext_s['date'])
ax.plot(ext_s['date'], ext_s['weather_index'], color='#3498DB', label='Weather', linewidth=1.8)
ax.plot(ext_s['date'], ext_s['traffic_index'], color='#E67E22', label='Traffic', linewidth=1.8)
ax.set_title('Daily External Factors (April 2026)')
ax.set_ylabel('Index (0=mild, 1=severe)')
ax.legend()
ax.tick_params(axis='x', rotation=30)
ax = axes[1, 1]
ax.hist(df['delay_ratio'], bins=40, color=palette['accent'], edgecolor='white', alpha=0.85)
ax.axvline(1.2, color=palette['delayed'], linestyle='--', linewidth=2, label='Threshold (1.2x)')
ax.axvline(df['delay_ratio'].mean(), color=palette['dark'], linestyle=':', linewidth=1.5,
           label=f'Mean ({df["delay_ratio"].mean():.2f}x)')
ax.set_title('Delay Ratio Distribution')
ax.set_xlabel('Actual / Expected Time')
ax.set_ylabel('Count')
ax.legend(fontsize=8)
ax = axes[1, 2]
df['date_dt'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date_dt'].dt.dayofweek
dow_delay = df.groupby('day_of_week')['delay_flag'].mean().reindex(range(7)).fillna(0)
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
ax.bar(days, dow_delay * 100, color=palette['accent'], edgecolor='white', linewidth=0.8, width=0.6)
ax.set_title('Delay Rate by Day of Week')
ax.set_ylabel('Delay Rate (%)')
ax.set_ylim(85, 105)
plt.tight_layout()
plt.savefig('plots/eda_dashboard.png', bbox_inches='tight', dpi=140)
plt.close()
print("\nEDA plot saved to plots/eda_dashboard.png")
