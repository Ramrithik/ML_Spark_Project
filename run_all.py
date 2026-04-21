import subprocess, sys, time
steps = [
    ('01_eda.py',                 'Step 1: EDA'),
    ('02_feature_engineering.py', 'Step 2: Feature Engineering'),
    ('03_model_training.py',      'Step 3: Model Training'),
    ('04_reward_optimization.py', 'Step 4: Reward Optimization'),
]
for script, label in steps:
    print(f"\n{'='*55}")
    print(f"  Running {label} ...")
    print(f"{'='*55}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed. Aborting.")
        sys.exit(1)
    print(f"\n[DONE] {label} ({elapsed:.1f}s)")

print("\n" + "="*55)
print("  ALL STEPS COMPLETE")
print("  Plots saved in: ./plots/")
print("  Model saved  : model_artifacts.pkl")
print("="*55)
