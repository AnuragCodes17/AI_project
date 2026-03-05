# -*- coding: utf-8 -*-
"""
# 🏎️ F1 Race Finish Prediction
Predicting a driver's finishing outcome using classification models.

**Dataset:** Kaggle Formula 1 World Championship (2019–2024)  
**Target:** 3 classes — `Podium 🏆` | `Points ✅` | `No Points / DNF ❌`  
**Features:** Grid position, rolling averages, qualifying gap to pole, championship standings
"""

## 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

print('Libraries imported successfully!')

"""## 2. Load Data"""

results      = pd.read_csv('data/results.csv')
races        = pd.read_csv('data/races.csv')
drivers      = pd.read_csv('data/drivers.csv')
constructors = pd.read_csv('data/constructors.csv')
qualifying   = pd.read_csv('data/qualifying.csv')
standings    = pd.read_csv('data/driver_standings.csv')

print('Results shape:    ', results.shape)
print('Qualifying shape: ', qualifying.shape)
print('Standings shape:  ', standings.shape)
results.head()

"""## 3. Exploratory Data Analysis"""

print(results.info())
print('\nMissing values:', results.isnull().sum().sum())

print(results[['grid', 'positionOrder', 'driverId', 'constructorId']].describe())

"""## 4. Data Preprocessing & Feature Engineering"""

#filter and sort
races_filtered   = races[(races['year'] >= 2019) & (races['year'] <= 2024)][['raceId', 'year', 'round']]
results_filtered = results.merge(races_filtered, on='raceId', how='inner')

data = results_filtered[['raceId', 'grid', 'driverId', 'constructorId', 'positionOrder', 'statusId']].copy()
data = data[data['grid'] > 0].dropna()
data = data.merge(races[['raceId', 'year', 'round']], on='raceId', how='left')
data = data.sort_values(['year', 'round']).reset_index(drop=True)

#rolling features
fallback = data['positionOrder'].mean()

data['driver_rolling_avg']      = data.groupby('driverId')['positionOrder'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(fallback)
data['constructor_rolling_avg'] = data.groupby('constructorId')['positionOrder'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(fallback)
data['driver_season_avg']       = data.groupby(['driverId', 'year'])['positionOrder'].transform(
    lambda x: x.shift(1).expanding().mean()).fillna(fallback)

high_grid_avg               = data[data['grid'] >= 15].groupby('driverId')['positionOrder'].mean()
data['driver_highgrid_avg'] = data['driverId'].map(high_grid_avg).fillna(fallback)

#qualifying and standings
def best_q_time(row):
    for col in ['q3', 'q2', 'q1']:
        if pd.notna(row[col]) and str(row[col]) != '\\N':
            return row[col]
    return np.nan

def to_seconds(t):
    try:
        m, s = str(t).split(':')
        return int(m) * 60 + float(s)
    except:
        return np.nan

qualifying['best_q_seconds'] = qualifying.apply(best_q_time, axis=1).apply(to_seconds)
pole_times                   = qualifying.groupby('raceId')['best_q_seconds'].min().rename('pole_time')
qualifying                   = qualifying.join(pole_times, on='raceId')
qualifying['q_gap_to_pole']  = qualifying['best_q_seconds'] - qualifying['pole_time']

data = data.merge(qualifying[['raceId', 'driverId', 'q_gap_to_pole']], on=['raceId', 'driverId'], how='left')
data['q_gap_to_pole'] = data['q_gap_to_pole'].fillna(data['q_gap_to_pole'].median())

data = data.merge(
    standings[['raceId', 'driverId', 'points', 'position']].rename(columns={'points': 'champ_points', 'position': 'champ_position'}),
    on=['raceId', 'driverId'], how='left'
)
data['champ_points']   = data['champ_points'].fillna(0)
data['champ_position'] = data['champ_position'].fillna(20)

data = data.dropna()
print('Dataset shape:', data.shape)
data.head()

"""## 5. Define Target & Split"""

def classify_finish(row):
    if row['statusId'] > 13:        return 2  # DNF
    elif row['positionOrder'] <= 3:  return 0  # Podium
    elif row['positionOrder'] <= 10: return 1  # Points
    else:                            return 2  # No Points

y = data.apply(classify_finish, axis=1)

label_names = {0: 'Podium 🏆', 1: 'Points ✅', 2: 'No Points / DNF ❌'}
print('Class distribution:')
print(y.value_counts().rename(label_names).sort_index())

preprocessor = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'), ['driverId', 'constructorId'])
], remainder='passthrough')

X = data[['grid', 'driverId', 'constructorId',
           'driver_rolling_avg', 'constructor_rolling_avg',
           'driver_highgrid_avg', 'driver_season_avg',
           'q_gap_to_pole', 'champ_points', 'champ_position']]
X = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('\nTraining set size:', X_train.shape)
print('Test set size:    ', X_test.shape)

"""## 6. Train Models"""

from sklearn.model_selection import RandomizedSearchCV

log_reg = LogisticRegression(max_iter=5000, class_weight='balanced')
log_reg.fit(X_train, y_train)
print('Logistic Regression trained!')

dt_clf = DecisionTreeClassifier(random_state=42, max_depth=8, min_samples_leaf=2, class_weight='balanced')
dt_clf.fit(X_train, y_train)
print('Decision Tree trained!')

rf_clf = RandomForestClassifier(n_estimators=200, max_depth=15, max_features='sqrt', random_state=42, class_weight='balanced')
rf_clf.fit(X_train, y_train)
print('Random Forest trained!')

xgb_search = RandomizedSearchCV(
    XGBClassifier(random_state=0),
    {
        'n_estimators':     [100, 200, 300, 500],
        'max_depth':        [3, 4, 5, 6, 8],
        'learning_rate':    [0.01, 0.05, 0.1, 0.2],
        'subsample':        [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
    },
    n_iter=30, cv=3, scoring='f1_weighted', random_state=42, n_jobs=1
)
xgb_search.fit(X_train, y_train)
xgb_clf = xgb_search.best_estimator_
print('Best XGBoost params:', xgb_search.best_params_)
print('XGBoost tuned!')

models = [
    ('Logistic Regression', log_reg),
    ('Decision Tree',       dt_clf),
    ('Random Forest',       rf_clf),
    ('XGBoost',             xgb_clf),
]

"""## 7. Model Evaluation"""

def evaluate_clf(name, y_test, y_pred):
    return {
        'Model':    name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 3),
        'F1 Score': round(f1_score(y_test, y_pred, average='weighted'), 3),
    }

eval_results = pd.DataFrame([
    evaluate_clf(name, y_test, model.predict(X_test)) for name, model in models
])

print(eval_results.to_string(index=False))

print('XGBoost — Classification Report')
print('-' * 50)
print(classification_report(y_test, xgb_clf.predict(X_test),
      target_names=['Podium', 'Points', 'No Points / DNF']))

"""## 8. Visualisations"""

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(x='Model', y='Accuracy', data=eval_results, palette='viridis', hue='Model', legend=False, ax=axes[0])
axes[0].set_title('Model Comparison — Accuracy')
axes[0].set_ylim(0.4, 0.85)
axes[0].tick_params(axis='x', rotation=15)

sns.barplot(x='Model', y='F1 Score', data=eval_results, palette='magma', hue='Model', legend=False, ax=axes[1])
axes[1].set_title('Model Comparison — F1 Score (Weighted)')
axes[1].set_ylim(0.4, 0.85)
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()

class_labels = ['Podium', 'Points', 'No Points / DNF']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, (name, model) in zip(axes.flatten(), models):
    cm   = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(name)
    ax.tick_params(axis='x', rotation=15)

plt.suptitle('Confusion Matrices — All Models', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

importances   = xgb_clf.feature_importances_
feature_names = ['grid', 'driver_rolling_avg', 'constructor_rolling_avg',
                  'driver_highgrid_avg', 'driver_season_avg',
                  'q_gap_to_pole', 'champ_points', 'champ_position']
numeric_importances = importances[-len(feature_names):]

plt.figure(figsize=(9, 5))
sns.barplot(x=numeric_importances, y=feature_names, palette='coolwarm', hue=feature_names, legend=False)
plt.title('XGBoost — Numeric Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

"""## 9. Predict a Single Driver"""

driver_names      = dict(zip(drivers['driverId'], drivers['surname']))
constructor_names = dict(zip(constructors['constructorId'], constructors['name']))

# ── Change these values ──
grid_position  = 20
driver_id      = 830
constructor_id = 9
# ─────────────────────────

def get_features(driver_id, constructor_id):
    driver_data = data[data['driverId'] == driver_id].sort_values(['year', 'round'])
    cons_data   = data[data['constructorId'] == constructor_id].sort_values(['year', 'round'])
    mean_pos    = data['positionOrder'].mean()

    rolling  = driver_data['positionOrder'].tail(5).mean()
    cons_rol = cons_data['positionOrder'].tail(5).mean()
    season   = driver_data['positionOrder'].mean()

    return (
        rolling  if not np.isnan(rolling)  else mean_pos,
        cons_rol if not np.isnan(cons_rol) else mean_pos,
        high_grid_avg.get(driver_id, mean_pos),
        season   if not np.isnan(season)   else mean_pos,
        driver_data['q_gap_to_pole'].iloc[-1]  if len(driver_data) > 0 else data['q_gap_to_pole'].median(),
        driver_data['champ_points'].iloc[-1]   if len(driver_data) > 0 else 0,
        driver_data['champ_position'].iloc[-1] if len(driver_data) > 0 else 20,
    )

rolling_avg, cons_rolling, driver_hg, season_avg, q_gap, champ_pts, champ_pos = get_features(driver_id, constructor_id)

print(f'Driver:        {driver_names.get(driver_id, driver_id)}')
print(f'Constructor:   {constructor_names.get(constructor_id, constructor_id)}')
print(f'Grid Position: {grid_position}')

input_df = pd.DataFrame(
    [[grid_position, driver_id, constructor_id, rolling_avg, cons_rolling,
      driver_hg, season_avg, q_gap, champ_pts, champ_pos]],
    columns=['grid', 'driverId', 'constructorId', 'driver_rolling_avg', 'constructor_rolling_avg',
             'driver_highgrid_avg', 'driver_season_avg', 'q_gap_to_pole', 'champ_points', 'champ_position']
)
input_data     = preprocessor.transform(input_df)
outcome_map    = {0: 'Podium 🏆', 1: 'Points Finish ✅', 2: 'No Points / DNF ❌'}
position_range = {0: '(P1–P3)', 1: '(P4–P10)', 2: '(P11+/DNF)'}
results_list   = []

for name, model in models:
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][pred]
    results_list.append({'Model': name, 'pred': pred, 'prob': prob})

vote_counts   = Counter(r['pred'] for r in results_list)
top_votes     = vote_counts.most_common()
majority_pred = top_votes[0][0]
is_tied       = len(top_votes) > 1 and top_votes[0][1] == top_votes[1][1]
best          = max(results_list, key=lambda r: r['prob'])

print()
for r in results_list:
    marker = '  ◀ most confident' if r['Model'] == best['Model'] else ''
    print(f"{r['Model']:22s} -> {outcome_map[r['pred']]:25s} {position_range[r['pred']]:12s} (confidence: {r['prob']:.2%}){marker}")

print(f"\n{'─'*60}")
if is_tied:
    print(f"  Majority verdict:    Tied — see most confident model")
else:
    print(f"  Majority verdict:    {outcome_map[majority_pred]} {position_range[majority_pred]}")
print(f"  Most confident:      {best['Model']} → {outcome_map[best['pred']]} {position_range[best['pred']]} at {best['prob']:.2%}")
print(f"{'─'*60}")

"""## 10. Model Accuracy Summary"""

print('Model Accuracy Summary:')
print('-' * 50)
for _, row in eval_results.iterrows():
    print(f"{row['Model']:22s} | Accuracy: {row['Accuracy']} | F1: {row['F1 Score']}")

best_model = eval_results.loc[eval_results['F1 Score'].idxmax()]
print(f"\nBest Model: {best_model['Model']}")
print(f"Best F1:    {best_model['F1 Score']}")
print(f"Best Acc:   {best_model['Accuracy']}")