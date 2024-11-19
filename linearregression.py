import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import pandas as pd

# Load the dataset
df = pd.read_csv('/kaggle/input/tech-olympiad-2024-bahrain-zain-challenge/TrainData.csv', delimiter='|')  # Replace with your actual data file

features = ['CO(GT)','PT08.S1(CO)','PT08.S2(NMHC)','PT08.S2_NMHC_to_PT08.S1_CO_Ratio','NMHC(GT)','NOx(GT)','PT08.S3(NOx)','NO2(GT)','PT08.S4(NO2)','PT08.S5(O3)']  # Replace with actual columns
target = 'RH'  # Replace with the actual target column name

# Prepare the features and target
X = df[features]
y = df[target]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (optional but can improve performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load your datasets
train_data = pd.read_csv('/kaggle/input/tech-olympiad-2024-bahrain-zain-challenge/TrainData.csv', delimiter='|')  # Replace with your actual file paths
test_data = pd.read_csv('/kaggle/input/tech-olympiad-2024-bahrain-zain-challenge/TestData.csv', delimiter='|')

# Verify column names
print("Train Data Columns:", train_data.columns)
print("Test Data Columns:", test_data.columns)

# Convert string numbers to numeric (if needed)
def convert_to_numeric(df):
    for column in df.columns:
        if df[column].dtype == 'object':  # Check if the column is of object type (strings)
            # Replace commas with dots and convert to numeric
            df[column] = df[column].str.replace(',', '.', regex=False)
            df[column] = pd.to_numeric(df[column], errors='coerce')  # Convert to numbers, set errors to NaN
    return df

train_data = convert_to_numeric(train_data)
test_data = convert_to_numeric(test_data)

# Handle missing values in 'RH' (target column)
# Option 1: Impute missing values in y_train (target column)
y_train = train_data['RH']
imputer_target = SimpleImputer(strategy='median')
y_train = imputer_target.fit_transform(y_train.values.reshape(-1, 1))  # Reshape y_train to 2D array

# Option 2: Alternatively, drop rows with missing target values (if you prefer this approach)
# train_data = train_data.dropna(subset=['RH'])
# y_train = train_data['RH']

# Separate features and target in train_data (after handling missing target values)
X_train = train_data.drop(columns=['RH'])  # Drop target column

# Use test_data as features only (it doesn't have RH)
X_test = test_data

# Handle missing values in features
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)  # Impute missing values in training data
X_test = imputer.transform(X_test)        # Impute missing values in testing data

# Standardize features (optional)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # Standardize training features
X_test = scaler.transform(X_test)         # Standardize testing features

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)
print("Predictions on test data:", y_pred)


from sklearn.metrics import mean_squared_error

# Assuming you've trained the model, make predictions on the training set (since test data doesn't have the target 'RH')
y_train_pred = model.predict(X_train)

# Calculate Mean Squared Error on training data (since y_test doesn't exist in test_data)
mse_train = mean_squared_error(y_train, y_train_pred)
print(f'Mean Squared Error on training data: {mse_train}')

# Optionally, calculate R-squared (explained variance) on training data
r2_train = model.score(X_train, y_train)
print(f'R-squared on training data: {r2_train}')


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# Assuming 'train_data' and 'test_df' are already loaded
# And 'features' is a list of feature column names

# Separate features and target in train_data
X_train = train_data[features]
y_train = train_data['RH']  # Assuming 'RH' is the target column in train data

# Check and handle missing values in target variable (y_train)
y_imputer = SimpleImputer(strategy='mean')  # Impute missing values in y_train using mean
y_train = y_imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()  # Reshape and impute, then flatten

# Handle missing values in features (for both training and test data)
imputer = SimpleImputer(strategy='mean')  # Impute missing values in features
X_train = imputer.fit_transform(X_train)  # Apply imputer to training data
X_test_kaggle = test_df[features]
X_test_kaggle = imputer.transform(X_test_kaggle)  # Apply imputer to test data

# Standardize features (optional)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # Apply standardization to training data
X_test_kaggle = scaler.transform(X_test_kaggle)  # Apply standardization to test data

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the Kaggle test data
y_test_pred = model.predict(X_test_kaggle)

# Create a submission DataFrame (assuming the test set has an 'ID' column)
submission = pd.DataFrame({
    'ID': test_df['ID'],  # Replace with your test set's ID column
    'RH': y_test_pred
})

# Save the submission file in the required format
submission.to_csv('submission.csv', index=False)
