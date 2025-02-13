import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv("parkinsons_data.csv")

# Display the first few rows
print("Dataset Preview:")
print(data.head())

# Display dataset information
print("\nDataset Info:")
print(data.info())

# Display summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values in Dataset:")
print(data.isnull().sum())

# Drop the 'name' column (not needed)
data = data.drop(columns=['name'])

print("\nColumns after dropping 'name':")
print(data.columns)

# Normalizee data (excluding the target variable 'status')
scaler = MinMaxScaler()
features = data.drop(columns=['status'])  # Features (X)
target = data['status']  # Target (y)

scaled_features = scaler.fit_transform(features)
data_normalized = pd.DataFrame(scaled_features, columns=features.columns)

# Add back the target column
data_normalized['status'] = target

print("\nNormalized Data Preview:")
print(data_normalized.head())


# Plot histograms of all features
data_normalized.hist(figsize=(15, 12), bins=30, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# Compute correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data_normalized.corr(), cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Count the number of Parkinson’s vs. non-Parkinson’s cases
plt.figure(figsize=(6, 4))
sns.countplot(x=data_normalized['status'], palette="coolwarm")
plt.title("Class Distribution: Parkinson's vs. Healthy")
plt.xlabel("Status (0 = Healthy, 1 = Parkinson's)")
plt.ylabel("Count")
plt.show()

# Print actual counts
print("\nClass Distribution:")
print(data_normalized['status'].value_counts())

# Set a correlation threshold (e.g., 0.9 means very high correlation)
correlation_threshold = 0.9

# Compute the correlation matrix
corr_matrix = data_normalized.corr()

# Identify highly correlated features
high_corr_features = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > correlation_threshold:  # Check if correlation is high
            colname = corr_matrix.columns[i]
            high_corr_features.add(colname)

print("\nHighly Correlated Features to Remove:", high_corr_features)

# Drop the highly correlated features
data_selected = data_normalized.drop(columns=high_corr_features)

print("\nRemaining Features After Correlation Filtering:")
print(data_selected.columns)

# Separate features (X) and target (y)
X = data_selected.drop(columns=['status'])  # Features
y = data_selected['status']  # Target variable

# Train a simple Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Get feature importance scores
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)

# Sort features by importance
important_features = feature_importances.sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=important_features.values, y=important_features.index, palette="coolwarm")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance using Random Forest")
plt.show()

# Keep only the most important features (top 10)
selected_features = important_features[:10].index
X_selected = X[selected_features]

print("\nTop 10 Selected Features:", selected_features)

from sklearn.model_selection import train_test_split

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)  # Train model
    y_pred = model.predict(X_test)  # Make predictions
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    
    print(f"\n{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))  # Print detailed metrics

import matplotlib.pyplot as plt

# Store accuracy scores
model_accuracies = {}

for name, model in models.items():
    y_pred = model.predict(X_test)  # Predict on test data
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    model_accuracies[name] = accuracy  # Store accuracy

# Plot model accuracies
plt.figure(figsize=(8, 5))
plt.bar(model_accuracies.keys(), model_accuracies.values(), color=['blue', 'green', 'red', 'purple'])
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")
plt.title("Model Performance Comparison")
plt.ylim(0, 1)  # Ensure the y-axis ranges from 0 to 1
plt.xticks(rotation=20)
plt.show()

# Select the best model (Modify this based on your results)
best_model = KNeighborsClassifier(n_neighbors=5)
best_model.fit(X_train, y_train)

# Save the trained model
import joblib
joblib.dump(best_model, "parkinsons_model.pkl")

print("\nBest Model Saved as 'parkinsons_model.pkl'")

