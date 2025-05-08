import joblib
import pandas as pd
import ipaddress

# Load the model
model = joblib.load(r'C:\Users\admin\Desktop\AI ThreatSense Platform\trained_models\trained_model.pkl')

# New data (example IP address)
new_data = pd.DataFrame([{
    'ip_address': '103.25.123.11',  # Example IP address
    'threat_score': 65,
    'alert_hour': 3,
    'alert_day': 2
}])

# Convert IP address to numeric format
new_data['ip_numeric'] = new_data['ip_address'].apply(lambda x: int(ipaddress.IPv4Address(x)))

# Prepare data for prediction
X_new = new_data[['ip_numeric', 'threat_score', 'alert_hour', 'alert_day']]

# Make prediction
prediction = model.predict(X_new)
print("Prediction for new data:", prediction)
