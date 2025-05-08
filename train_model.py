import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import ipaddress
import joblib

# Load the dataset
df = pd.read_csv('C:/Users/admin/Desktop/AI ThreatSense Platform/dataset/threat_data.csv')

# Convert IP address to numeric (using the ipaddress library)
df['ip_numeric'] = df['ip_address'].apply(lambda x: int(ipaddress.IPv4Address(x)))

# Drop the original ip_address column
df = df.drop('ip_address', axis=1)

# Features and target variable
X = df[['ip_numeric', 'threat_score', 'alert_hour', 'alert_day']]
y = df['label']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_train_upsampled, y_train_upsampled = smote.fit_resample(X_train, y_train)

# Initialize Logistic Regression classifier
model = LogisticRegression(random_state=42)

# Train the model
model.fit(X_train_upsampled, y_train_upsampled)

# Evaluate the model
y_pred = model.predict(X_test)
print("Model Accuracy on Test Set:", model.score(X_test, y_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'trained_models/trained_model.pkl')
print("Model saved as 'trained_model.pkl'")
