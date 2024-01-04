import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

pricing_data_df = pd.read_csv('pricing_data_without_CITY12142023.csv')

print(pricing_data_df)


print(pricing_data_df.shape)

req_features = ['SCOPE_OF_WORK_PROJECT_ID',
                'CLIENT_RATE_OVERRIDE',
'PROJECT_SIGNED',
'PROJECT_TASK_COUNT',
'SERVICE_CFO',
'SERVICE_CPA/Accounting Advisory',
'SERVICE_FP&A',
'SERVICE_Full Charge Bookkeeping',
'SERVICE_Tax Preparation',
'PROJECT_FREQUENCY_One Time',
'PROJECT_FREQUENCY_Recurring monthly',
'BILL_RATE_TYPE_Fixed',
'BILL_RATE_TYPE_Hourly']

data_2= pricing_data_df[req_features]



X = data_2.drop(['CLIENT_RATE_OVERRIDE','SCOPE_OF_WORK_PROJECT_ID'], axis=1)
y = data_2['CLIENT_RATE_OVERRIDE']

print(X)
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    feature_importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print("Feature Importances:")
    print(feature_importance_df)
except Exception as e:
    print(e)

import pandas as pd
df = pd.DataFrame(feature_importance_df)
print("df")

df.to_csv('feature_importance.csv', index=False)

top_n = 5
sorted_feature_importances = feature_importance_df.sort_values(by='Importance', ascending=False)
top_features = sorted_feature_importances.head(top_n)['Feature'].tolist()
print(f"Top {top_n} Features:")
print(top_features)

X = pricing_data_df[top_features] 
y = pricing_data_df['CLIENT_RATE_OVERRIDE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

