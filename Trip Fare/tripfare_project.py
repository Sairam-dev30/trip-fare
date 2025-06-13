import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pytz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Load dataset
df = pd.read_csv(r"C:\Users\DELL\Downloads\taxi_fare.csv")
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], utc=True)

# Step 2: Feature Engineering
eastern = pytz.timezone('US/Eastern')
df['tpep_pickup_datetime'] = df['tpep_pickup_datetime'].dt.tz_convert(eastern)
df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
df['pickup_day'] = df['tpep_pickup_datetime'].dt.day_name()
df['is_night'] = df['pickup_hour'].apply(lambda x: 1 if (x < 6 or x > 22) else 0)
df['is_weekend'] = df['pickup_day'].isin(['Saturday', 'Sunday']).astype(int)
df['am_pm'] = df['pickup_hour'].apply(lambda x: 'AM' if x < 12 else 'PM')

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

df['trip_distance'] = haversine_distance(
    df['pickup_latitude'], df['pickup_longitude'],
    df['dropoff_latitude'], df['dropoff_longitude']
)

# Step 3: Drop datetime column
df.drop(columns=['tpep_pickup_datetime'], inplace=True, errors='ignore')

# Step 4: Data Cleaning
df = df.dropna()
df = df[(df['trip_distance'] > 0) & (df['total_amount'] > 0)]

# Step 5: Outlier Removal
def remove_outliers_iqr(col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

df = remove_outliers_iqr('total_amount')
df = remove_outliers_iqr('trip_distance')

# Step 6: Fix Skewness
df['trip_distance'] = np.log1p(df['trip_distance'])
df['total_amount'] = np.log1p(df['total_amount'])

# Step 7: Encode Categorical Variables
df = pd.get_dummies(df, columns=['pickup_day', 'am_pm'], drop_first=True)

# ✅ Step 8: Final cleanup for modeling
# Drop any remaining non-numeric columns
df = df.select_dtypes(include=[np.number])

# Define X and y
X_all = df.drop(columns=['total_amount'])
y = df['total_amount']

# Final sanity check
print("❗ Non-numeric columns in features:", X_all.select_dtypes(include='object').columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)

# Step 9: Feature Importance
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
feature_importances = pd.DataFrame({
    'Feature': X_all.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n🔍 Top Features by Random Forest:")
print(feature_importances)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()

# Use top 7 features
top_features = feature_importances['Feature'].head(7).tolist()
X_train = X_train[top_features]
X_test = X_test[top_features]

# Step 10: Train 5 Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

best_model = None
best_r2 = -np.inf

print("\n📊 Model Performance:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    print(f"📌 {name}")
    print(f"R² Score       : {r2:.4f}")
    print(f"RMSE (Error)   : {rmse:.2f}")
    print(f"MAE  (Error)   : {mae:.2f}\n")

    if r2 > best_r2:
        best_model = model
        best_model_name = name
        best_r2 = r2

# Step 11: Save Best Model
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"✅ Best model saved: {best_model_name}")

# Step 12: Optional EDA
df['fare_per_km'] = np.expm1(df['total_amount']) / np.expm1(df['trip_distance'])

plt.figure(figsize=(8, 5))
sns.histplot(np.expm1(df['trip_distance']), bins=50, kde=True)
plt.title("Trip Distance Distribution")
plt.xlabel("Distance (km)")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x=np.expm1(df['trip_distance']), y=df['fare_per_km'], alpha=0.5)
plt.title("Fare per KM vs Trip Distance")
plt.xlabel("Trip Distance (km)")
plt.ylabel("Fare per KM")
plt.show()
  