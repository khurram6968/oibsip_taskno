# Importing necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset from CSV
file_path = "E:\datascience"
os.chdir(file_path)
iris_data = pd.read_csv("iris.csv")

# Assuming the last column is the target variable (species) and the rest are features
X = iris_data.drop("Species",axis=1)
y = iris_data["Species"]

# Class distribution
print("\nClass Distribution:")
print(y.value_counts())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Check for Overfitting

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the decision tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Calculate accuracy on both training and test sets
train_accuracy = accuracy_score(y_train, dt.predict(X_train))
test_accuracy = accuracy_score(y_test, dt.predict(X_test))

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'max_depth': [3, 4, 5, 6],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [2, 3, 4]
}
dtc = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dtc, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters and cross-validation score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"\nBest Parameters: {best_params}")
print(f"Best Cross-Validation Score: {best_score:.4f}")

# Train final model with best parameters
best_model = DecisionTreeClassifier(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# Predictions and evaluation on the test set
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}\n")
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris_data['Species'].unique(), yticklabels=iris_data['Species'].unique())
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature importance
feature_importances = best_model.feature_importances_
features = X.columns

# Plot feature importances
plt.figure(figsize=(8, 6))
plt.barh(features, feature_importances, color='skyblue')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importances in the Best Decision Tree")
plt.show()

# Decision Tree Plot
plt.figure(figsize=(12, 8))
plot_tree(best_model, feature_names=X.columns, class_names=y.unique(), filled=True)
plt.title("Decision Tree Visualization")
plt.show()