import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Data Generation (Synthetic Data for Demonstration) ---
# In a real scenario, you would load your 'date,device,failure,metric1,metric2, ...' dataset here.
# print("Generating synthetic data...")
# np.random.seed(42)
# num_devices = 50
# days_per_device = 365
# dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=days_per_device))

# data = []
# for i in range(1, num_devices + 1):
#     device_id = f'DEV{i:03d}'
#     # Simulate normal operation with some random fluctuations
#     metric1 = np.random.normal(50, 5, days_per_device).cumsum() / 10 + np.random.normal(0, 1, days_per_device)
#     metric2 = np.random.normal(100, 10, days_per_device).cumsum() / 15 + np.random.normal(0, 2, days_per_device)
#     metric3 = np.random.normal(20, 2, days_per_device)
#     metric4 = np.random.normal(5, 1, days_per_device)
#     metric5 = np.random.normal(150, 15, days_per_device)

#     # Simulate some failures for a few devices with increasing metrics before failure
#     failure = np.zeros(days_per_device, dtype=int)
#     if i % 5 == 0: # Every 5th device has a failure
#         failure_day_index = np.random.randint(50, days_per_device - 10) # Failure after some days
#         failure[failure_day_index] = 1
#         # Increase metrics leading up to failure for more realistic simulation
#         metric1[failure_day_index-10:failure_day_index+1] = np.linspace(metric1[failure_day_index-10], metric1[failure_day_index-10] + 30, 11)
#         metric2[failure_day_index-10:failure_day_index+1] = np.linspace(metric2[failure_day_index-10], metric2[failure_day_index-10] + 50, 11)


#     for d_idx, day in enumerate(dates):
#         data.append([
#             day,
#             device_id,
#             failure[d_idx],
#             metric1[d_idx],
#             metric2[d_idx],
#             metric3[d_idx],
#             metric4[d_idx],
#             metric5[d_idx]
#         ])

df = pd.read_csv('Python/predictive_maintenance_dataset.csv')

print(f"Synthetic data generated. Shape: {df.shape}")
print("First 5 rows of the synthetic data:")
print(df.head())
print(f"Failure count: {df['failure'].sum()} ({(df['failure'].sum()/len(df)):.2%} of total data)")

# --- 2. Data Preprocessing and Feature Engineering ---
print("\nPerforming feature engineering...")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['device', 'date']).reset_index(drop=True)

# Define metrics columns to create features from
metric_cols = [col for col in df.columns if 'metric' in col]

# Create lagged features (e.g., previous 1, 3, 7 days)
for col in metric_cols:
    for lag in [1, 3, 7]:
        df[f'{col}_lag_{lag}'] = df.groupby('device')[col].shift(lag)

# Create rolling window features (e.g., mean and std over last 7 days)
for col in metric_cols:
    for window in [7, 14]:
        df[f'{col}_rolling_mean_{window}d'] = df.groupby('device')[col].rolling(window=window).mean().reset_index(level=0, drop=True)
        df[f'{col}_rolling_std_{window}d'] = df.groupby('device')[col].rolling(window=window).std().reset_index(level=0, drop=True)

# Drop rows with NaN values introduced by lagging/rolling (typically at the start of each device's history)
df_processed = df.dropna()
print(f"Data after feature engineering and dropping NaNs. Shape: {df_processed.shape}")
print("First 5 rows of processed data:")
print(df_processed.head())

# --- 3. Define Features (X) and Target (y) ---
# The target variable
y = df_processed['failure']

# Features (all columns except 'date', 'device', 'failure')
X = df_processed.drop(['date', 'device', 'failure'], axis=1)

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print("Example features (first row):")
print(X.iloc[0])

# --- 4. Data Splitting ---
# It's important to ensure that the train and test sets are representative.
# For time series data, a time-based split is often preferred to avoid data leakage,
# where you train on older data and test on newer data.
# For simplicity, we'll use a random split here, but for real-world deployment, consider:
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) # if data is already sorted by time
# Or split by date:
# split_date = pd.to_datetime('2023-10-01')
# X_train = X[df_processed['date'] < split_date]
# y_train = y[df_processed['date'] < split_date]
# X_test = X[df_processed['date'] >= split_date]
# y_test = y[df_processed['date'] >= split_date]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=42, stratify=y)
print(f"\nTraining set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")
print(f"Failure in training set: {y_train.sum()} ({(y_train.sum()/len(y_train)):.2%} of training data)")
print(f"Failure in test set: {y_test.sum()} ({(y_test.sum()/len(y_test)):.2%} of test data)")


# --- 5. Model Training (XGBoost) ---
print("\nTraining XGBoost model...")

# Calculate scale_pos_weight to handle class imbalance
# This parameter helps the model focus more on correctly classifying the minority class (failures)
positive_cases = y_train.sum()
negative_cases = len(y_train) - positive_cases
scale_pos_weight_value = negative_cases / positive_cases if positive_cases > 0 else 1

print(f"Calculated scale_pos_weight: {scale_pos_weight_value:.2f}")

# Initialize XGBoost Classifier
# Parameters are tuned to address common issues in classification:
# - objective='binary:logistic': for binary classification with probability output
# - eval_metric='logloss': evaluation metric during training
# - use_label_encoder=False: suppresses a warning about a deprecated parameter (for older XGBoost versions)
# - tree_method='hist': faster histogram-based tree construction
# - scale_pos_weight: crucial for imbalanced datasets
# - n_estimators: number of boosting rounds
# - learning_rate: step size shrinkage to prevent overfitting
# - max_depth: maximum depth of a tree
# - subsample: fraction of samples to be used for fitting the individual base learners
# - colsample_bytree: fraction of features to be used for fitting the individual base learners
# - gamma: minimum loss reduction required to make a further partition on a leaf node
# - random_state: for reproducibility
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    tree_method='hist', # Faster training
    scale_pos_weight=scale_pos_weight_value, # Crucial for imbalance
    n_estimators=300, # Number of boosting rounds
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)
print("Model training complete.")

# --- 6. Model Evaluation ---
print("\nEvaluating model performance...")

# Predict probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of failure (class 1)

# Predict binary outcomes using default threshold (0.5)
y_pred = (y_pred_proba >= 0.5).astype(int)

# Classification Report
print("\n--- Classification Report (Default Threshold 0.5) ---")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n--- Confusion Matrix (Default Threshold 0.5) ---")
print(cm)
# Interpretation:
# [[TN, FP],
#  [FN, TP]]
# TN: True Negatives (Correctly predicted non-failure)
# FP: False Positives (Incorrectly predicted failure - "False Alarm")
# FN: False Negatives (Incorrectly predicted non-failure - "Missed Failure")
# TP: True Positives (Correctly predicted failure)

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score: {roc_auc:.4f}")

# --- 7. Threshold Tuning for Balancing False Positives/Negatives ---
print("\nAnalyzing Precision-Recall Curve for Threshold Tuning...")

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='b', alpha=0.7, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall (True Positive Rate)')
plt.ylabel('Precision (Positive Predictive Value)')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()

# Find a threshold that balances precision and recall (e.g., where precision and recall are close, or based on specific business needs)
# You can iterate through thresholds to find the best balance.
# For predictive maintenance, often minimizing false negatives (missing a failure) is more important,
# even at the cost of some false positives (unnecessary maintenance).
# Let's find a threshold that aims for a certain recall (e.g., 80% recall, meaning we catch 80% of actual failures).
target_recall = 0.8
idx = np.where(recall >= target_recall)[0]
if len(idx) > 0:
    optimal_threshold_idx = idx[np.argmax(precision[idx])]
    optimal_threshold = thresholds[optimal_threshold_idx]
    print(f"\nSuggested threshold for ~{target_recall*100}% Recall: {optimal_threshold:.4f}")

    # Re-evaluate with the optimal threshold
    y_pred_tuned = (y_pred_proba >= optimal_threshold).astype(int)
    print(f"\n--- Classification Report (Threshold: {optimal_threshold:.4f}) ---")
    print(classification_report(y_test, y_pred_tuned))
    cm_tuned = confusion_matrix(y_test, y_pred_tuned)
    print("\n--- Confusion Matrix (Tuned Threshold) ---")
    print(cm_tuned)
else:
    print("\nCould not find a threshold achieving the target recall easily. Review the Precision-Recall curve.")

# --- 8. Feature Importance ---
print("\n--- Feature Importance ---")
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importances.head(10))

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importances.head(15))
plt.title('Top 15 Feature Importances (XGBoost)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

print("\nModel building and evaluation complete. Consider deploying this model or iterating on features/hyperparameters.")
