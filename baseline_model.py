import pandas as pd
import numpy as np
import os
import urllib.request
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Define columns
columns = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + ['s_' + str(i) for i in range(1,22)]

# Load data
print("Loading data...")
train = pd.read_csv('data/train_FD001.txt', sep='\s+', header=None, names=columns)
test = pd.read_csv('data/test_FD001.txt', sep='\s+', header=None, names=columns)
y_test = pd.read_csv('data/RUL_FD001.txt', sep='\s+', header=None, names=['RUL'])

# Calculate RUL for training data
def add_rul(df):
    max_cycle = df.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycle.columns = ['unit_nr', 'max_cycle']
    df = df.merge(max_cycle, on=['unit_nr'], how='left')
    df['RUL'] = df['max_cycle'] - df['time_cycles']
    df.drop('max_cycle', axis=1, inplace=True)
    return df

train = add_rul(train)

# Feature selection (dropping settings and constant sensors for FD001)
drop_cols = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3', 
             's_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
features = [c for c in train.columns if c not in drop_cols and c != 'RUL']

X_train = train[features]
y_train = train['RUL']

# For test set, we only predict the RUL at the LAST time cycle for each engine
X_test = test.groupby('unit_nr').last().reset_index()[features]

# Train Baseline Model (Random Forest)
print("Training Random Forest Baseline...")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Predict and Evaluate
preds = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test['RUL'], preds))

print(f"\n======================================")
print(f"Baseline Random Forest RMSE: {rmse:.2f}")
print(f"======================================\n")

# Print Top 5 Important Features
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
print("Top 5 Important Sensors (Baseline Feature Importance):")
for i in range(5):
    print(f"Sensor {features[indices[i]].replace('s_', '')}: {importances[indices[i]]:.4f}")
