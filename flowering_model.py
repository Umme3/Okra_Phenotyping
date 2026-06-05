import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np


EXCEL_PATH = r"okra_veg.xlsx"
FLOWERING_FILE = r"first_flowering_dates.xlsx"



flowering = pd.read_excel(FLOWERING_FILE)
flowering['Date'] = pd.to_datetime(flowering['Date'])
flowering['Days_to_Flower'] = flowering['flowering day']

features5 = pd.read_excel(EXCEL_PATH, sheet_name="Sheet5")
features6 = pd.read_excel(EXCEL_PATH, sheet_name="Sheet6")
features = pd.concat([features5, features6], ignore_index=True)


features.columns = features.columns.str.strip()


features['Date'] = pd.to_datetime(features['Date'], errors='coerce')

features = features.merge(
    flowering[['Subject', 'Date']],
    on='Subject',
    suffixes=('', '_flower')
)

features_before = features[features['Date'] < features['Date_flower']]


features_summary = features_before.groupby('Subject').agg({
    'Temp (F)': ['mean', 'max', 'min', 'sum'],
    'CO2 (ppm)': 'mean',
    'Relative Humidity (%)': 'mean',
    'Luminous Flux (lux)': ['mean', 'sum'],
    'Soil Temperature (F)': ['mean', 'max', 'min'],
    'Soil PH': 'mean',
    'Soil moisture content (%)': ['mean', 'sum'],
    'Date': 'count'  
}).reset_index()

features_summary.columns = ['_'.join(col).strip('_') for col in features_summary.columns]


data = flowering.merge(features_summary, on='Subject')
print("Data ready for modeling:")


X = data.drop(columns=['Subject', 'Date', 'Days_to_Flower'])
y = data['Days_to_Flower']
subjects = data['Subject']


X_train, X_test, y_train, y_test, subj_train, subj_test = train_test_split(X, y, subjects, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
model.fit(X_train, y_train)
importances = model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance)
y_test_pred = model.predict(X_test)
print("Test MAE:", mean_absolute_error(y_test, y_test_pred))
print("Test R²:", r2_score(y_test, y_test_pred))

subjects = subj_test.values
actual = y_test.values
predicted = y_test_pred

x = np.arange(len(subjects)) 
width = 0.35  

plt.figure(figsize=(10, 5))

plt.bar(x - width/2, actual, width, label='Actual')

plt.bar(x + width/2, predicted, width, label='Predicted')

plt.xticks(x, subjects)
plt.ylabel("Days to Flower")
plt.xlabel("Subject")
plt.title("Actual vs Predicted Days to Flower per Subject")
plt.legend()

