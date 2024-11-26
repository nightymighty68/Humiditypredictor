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
y = y.values.reshape(-1, 1)  # Reshape y to 2D array if it's a 1D array
y = imputer.fit_transform(y).ravel()  # Apply imputer to target and flatten it back to 1D

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional, but helps with tree-based models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error on the test data
mse_test = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on test data: {mse_test}')

# Optionally, calculate R-squared (explained variance) on test data
r2_test = model.score(X_test, y_test)
print(f'R-squared on test data: {r2_test}')


# Ensure X_test_kaggle is a DataFrame (which it should be by default)
X_test_kaggle = test_data[features]

# Double-check shape of X_test_kaggle before imputation
print("Shape of X_test_kaggle before imputation:", X_test_kaggle.shape)

# Apply imputer to the test data (it expects the input as a 2D DataFrame or numpy array)
X_test_kaggle_imputed = imputer.fit_transform(X_test_kaggle)  # Fit on X_train and then apply to X_test_kaggle

# Double-check shape after imputation
print("Shape of X_test_kaggle after imputation:", X_test_kaggle_imputed.shape)

# Standardize the imputed test data
X_test_kaggle_imputed = scaler.transform(X_test_kaggle_imputed)

# Make predictions for Kaggle test set
y_test_pred_kaggle = model.predict(X_test_kaggle_imputed)

# Create a submission DataFrame (assuming the test set has an 'ID' column)
submission = pd.DataFrame({
    'ID': test_data['ID'],  # Replace with your test set's ID column
    'RH': y_test_pred_kaggle
})

# Save the submission file in the required format
submission.to_csv('submission.csv', index=False)

# Create a submission DataFrame (assuming the test set has an 'ID' column)
submission = pd.DataFrame({
    'ID': test_data['ID'],  # Replace with your test set's ID column
    'RH': y_test_pred_kaggle
})

# Ensure that the submission has 1873 rows
print("Shape of submission:", submission.shape)

# Save the submission file in the required format
submission.to_csv('submission.csv', index=False)
