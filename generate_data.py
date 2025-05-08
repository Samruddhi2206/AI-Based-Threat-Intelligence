import pandas as pd
import numpy as np
import random
import ipaddress

# Function to generate random IP addresses
def random_ip():
    return str(ipaddress.IPv4Address(random.randint(1, 2**32 - 1)))

# Function to generate synthetic data
def generate_synthetic_data(n=100):
    data = []
    for _ in range(n):
        ip_address = random_ip()
        threat_score = random.randint(30, 100)  # Threat score between 30 and 100
        alert_hour = random.randint(0, 23)  # Hour between 0 and 23
        alert_day = random.randint(1, 7)  # Day between 1 and 7 (e.g., days of the week)
        label = random.choice([0, 1])  # Randomly assigning label 0 or 1
        data.append([ip_address, threat_score, alert_hour, alert_day, label])

    # Creating a DataFrame
    df = pd.DataFrame(data, columns=['ip_address', 'threat_score', 'alert_hour', 'alert_day', 'label'])
    return df

# Generate 100 synthetic data points
synthetic_df = generate_synthetic_data(100)

# Save to CSV (optional)
synthetic_df.to_csv(r'C:\Users\admin\Desktop\AI ThreatSense Platform\dataset\threat_data.csv', index=False)

# Show the first few rows of the synthetic dataset
print(synthetic_df.head())
