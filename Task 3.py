# Import necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset from CSV
file_path = "E:\datascience"
os.chdir(file_path)
data = pd.read_csv("car data.csv")


data.drop(['Car_Name'], axis=1, inplace=True)  # Drop the 'Car_Name' column

data['Car_Age'] = 2024 - data['Year']  # Assuming current year is 2024
data.drop(['Year'], axis=1, inplace=True)  # Drop the 'Year' column

#Encode categorical variables
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
encoder = OneHotEncoder(drop='first')  # One-hot encode the categorical variables

# Prepare features (X) and target variable (y)
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder, categorical_features),  # One-hot encode categorical variables
        ('num', StandardScaler(), ['Present_Price', 'Driven_kms', 'Car_Age'])  # Scale numerical features
    ])

# Define the model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

# Train the model
pipeline.fit(X_train, y_train)

#Predict on test data
y_pred = pipeline.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nEvaluation Metrics:")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R2 Score: {r2}")

#Hyperparameter Tuning using GridSearchCV
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters from GridSearchCV
print(f"\nBest Parameters from GridSearchCV: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_}")

# Final evaluation on test data using the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

mae_best = mean_absolute_error(y_test, y_pred_best)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2_best = r2_score(y_test, y_pred_best)

print(f"\nBest Model Evaluation Metrics:")
print(f"Mean Absolute Error: {mae_best}")
print(f"Root Mean Squared Error: {rmse_best}")
print(f"R2 Score: {r2_best}")