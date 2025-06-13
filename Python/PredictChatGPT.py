import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load dataset
df = pd.read_csv('predictive_maintenance_dataset.csv')  # replace with actual path

# 2. Basic preprocessing
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['device', 'date'])

# Encode 'device' if it's categorical
if df['device'].dtype == 'object':
    le = LabelEncoder()
    df['device'] = le.fit_transform(df['device'])

# 3. Feature/target split
X = df.drop(columns=['failure', 'date'])
y = df['failure']

# 4. Handle class imbalance if needed
print(y.value_counts())  # if imbalance is high, consider class_weight or resampling

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 6. Model training with LightGBM
model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    class_weight='balanced',  # handles class imbalance
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=50,
    verbose=False
)

# 7. Evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
