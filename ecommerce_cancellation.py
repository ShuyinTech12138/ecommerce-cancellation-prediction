import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Load data
orders = pd.read_csv('olist_orders_dataset.csv')
items = pd.read_csv('olist_order_items_dataset.csv')
payments = pd.read_csv('olist_order_payments_dataset.csv')
customers = pd.read_csv('olist_customers_dataset.csv')

# Create target variable: 1 = cancelled, 0 = not cancelled
orders['cancelled'] = (orders['order_status'] == 'canceled').astype(int)
print("Cancellation rate:", orders['cancelled'].mean().round(3))
print("Total orders:", len(orders))

# Feature engineering from orders
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_approved_at'] = pd.to_datetime(orders['order_approved_at'])
orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])

# Approval delay in hours
orders['approval_delay_hours'] = (
    orders['order_approved_at'] - orders['order_purchase_timestamp']
).dt.total_seconds() / 3600

# Estimated delivery window in days
orders['estimated_delivery_days'] = (
    orders['order_estimated_delivery_date'] - orders['order_purchase_timestamp']
).dt.days

# Purchase hour and day of week
orders['purchase_hour'] = orders['order_purchase_timestamp'].dt.hour
orders['purchase_dayofweek'] = orders['order_purchase_timestamp'].dt.dayofweek

# Aggregate items per order
items_agg = items.groupby('order_id').agg(
    num_items=('order_item_id', 'count'),
    total_price=('price', 'sum'),
    avg_price=('price', 'mean'),
    total_freight=('freight_value', 'sum')
).reset_index()

# Aggregate payments per order
payments_agg = payments.groupby('order_id').agg(
    total_payment=('payment_value', 'sum'),
    num_installments=('payment_installments', 'max'),
    payment_type=('payment_type', 'first')
).reset_index()

# Encode payment type
le = LabelEncoder()
payments_agg['payment_type_encoded'] = le.fit_transform(payments_agg['payment_type'])

# Merge all
df = orders.merge(items_agg, on='order_id', how='left')
df = df.merge(payments_agg[['order_id', 'total_payment', 'num_installments', 'payment_type_encoded']], on='order_id', how='left')
df = df.merge(customers[['customer_id', 'customer_state']], on='customer_id', how='left')

# Encode state
df['state_encoded'] = le.fit_transform(df['customer_state'].fillna('unknown'))

# Select features
features = [
    'approval_delay_hours', 'estimated_delivery_days',
    'purchase_hour', 'purchase_dayofweek',
    'num_items', 'total_price', 'avg_price', 'total_freight',
    'total_payment', 'num_installments', 'payment_type_encoded',
    'state_encoded'
]

df_model = df[features + ['cancelled']].dropna()
print("Model dataset size:", len(df_model))

X = df_model[features]
y = df_model['cancelled']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("After SMOTE - class distribution:", y_train.value_counts().to_dict())

# Hyperparameter tuning with GridSearchCV
params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}

xgb = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
grid_search = GridSearchCV(xgb, params, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("\nBest params:", grid_search.best_params_)

# Evaluate
best_model = grid_search.best_estimator_
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC Score: {auc:.4f}")

y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
importance = pd.DataFrame({
    'feature': features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop Features:")
print(importance.head(10))
