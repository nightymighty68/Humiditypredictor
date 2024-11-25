import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb

# Load the dataset
df = pd.read_csv('tech-olympiad-2024-bahrain-zain-challenge/TrainData.csv', delimiter='|')

features = ['CO(GT)', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S2_NMHC_to_PT08.S1_CO_Ratio',
            'NMHC(GT)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)']
target = 'RH'  # Replace with the actual target column name

# Convert commas to dots for numeric columns if necessary and handle any other non-numeric values
def convert_to_numeric(df):
    for column in df.columns:
        if df[column].dtype == 'object':  # If the column is of object type (strings)
            # Replace commas with dots (if any)
            df[column] = df[column].str.replace(',', '.', regex=False)
            # Try to convert the column to numeric values, coerce errors to NaN (handles strings like '0,7')
            df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

# Apply the conversion to the training and testing data
df = convert_to_numeric(df)

# Handle missing values by applying median imputation
imputer = SimpleImputer(strategy='median')
df[features] = imputer.fit_transform(df[features])  # Apply imputer to features

# Prepare the features and target
X = df[features]
y = df[target]

# Handle missing values in target (y) using median imputation
y = y.reshape(-1, 1)  # Reshape y to 2D array if it's already a NumPy array
y = imputer.fit_transform(y).ravel()  # Apply imputer to target and flatten it back to 1D


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional, but helps with tree-based models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import lightgbm as lgb

# Define the LightGBM model
lgb_model = lgb.LGBMRegressor(objective='regression', random_state=42)

# Train the model
lgb_model.fit(X_train, y_train)

# Make predictions
y_pred = lgb_model.predict(X_test)

# Calculate Mean Squared Error on the test data
mse_test = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on test data: {mse_test}')

# Optionally, calculate R-squared (explained variance) on test data
r2_test = lgb_model.score(X_test, y_test)  # Use lgb_model instead of model
print(f'R-squared on test data: {r2_test}')


# Check the shape of X_train_scaled and train_data['RH']
print(f"Shape of X_train_scaled: {X_train_scaled.shape}")
print(f"Shape of train_data['RH']: {train_data['RH'].shape}")

# Ensure 'RH' is numeric and there are no missing values
print(f"Missing values in RH: {train_data['RH'].isna().sum()}")

# If there are any missing values in 'RH', handle them (if applicable)
if train_data['RH'].isna().sum() > 0:
    train_data['RH'] = train_data['RH'].fillna(train_data['RH'].median())

# Ensure that 'RH' is of numeric type (float or int)
train_data['RH'] = pd.to_numeric(train_data['RH'], errors='coerce')

# Re-check the shape of X_train_scaled and RH
print(f"Shape of X_train_scaled: {X_train_scaled.shape}")
print(f"Shape of train_data['RH']: {train_data['RH'].shape}")

# Now, fit the model
lgb_model = lgb.LGBMRegressor(objective='regression', random_state=42)
lgb_model.fit(X_train_scaled, train_data['RH'])

# Make predictions for  test set
y_test_pred_kaggle = lgb_model.predict(X_test_kaggle_scaled)

# Create a submission DataFrame (assuming the test set has an 'ID' column)
submission = pd.DataFrame({
    'ID': test_data['ID'],  # Replace with your test set's ID column
    'RH': y_test_pred_kaggle
})

# Save the submission file in the required format
submission.to_csv('submission.csv', index=False)
