#  ----------------------
# AUTHOR / Brian Cook
# CLASS / SWE452
# DATE / 2/3/2025
# -----------------------

from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load the dataset
iris = load_iris()

# Features (input data)
X = iris.data

# Target (output data)
y = iris.target

# Names of the features
feature_names = iris.feature_names

# Names of the target
target_names = iris.target_names

# Print the features and target names of the dataset
print("Features\n", feature_names)
print("Target\n", target_names)
print("First 5 Samples:\n", X[:5])

data = pd.DataFrame(iris.data, columns=feature_names)

# CHeck for missing values
print("\nMissing Values per feature\n", data.isnull().sum())

#check for duplicates
print("\nNumber of duplicated rows\n", data.duplicated().sum())

data.drop_duplicates(inplace=True)

# CHeck for missing values again
print("\nMissing Values per feature\n", data.isnull().sum())

#check for duplicates again
print("\nNumber of duplicated rows\n", data.duplicated().sum())


#! Data Preprocessing -> Standardize Features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFirst 5 rows of the scaled training data\n", X_train_scaled[:5])

#! Step 3: train and evaluate the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# generate a classification report
print("\nClassification Report\n", classification_report(y_test, y_pred, target_names=target_names))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()