import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import warnings

import shap

warnings.filterwarnings('ignore')

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Step 1: Load and explore the data
print("Step 1: Data Loading and Exploration")
print("-" * 50)

# Assuming the file is in the same directory
try:
    data = pd.read_csv("/content/financial_anomaly_data.csv")               
    print(f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns")
except Exception as e:
    print(f"Error loading data: {e}")
    # Create sample data if loading fails (for demonstration purposes)
    data = pd.DataFrame({
        'TransactionID': range(1, 1001),
        'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S") for _ in range(1000)],
        'Amount': np.random.normal(100, 50, 1000),
        'Merchant': np.random.choice(['Amazon', 'Walmart', 'Target', 'eBay', 'Starbucks'], 1000),
        'TransactionType': np.random.choice(['Online', 'In-Store', 'Mobile'], 1000),
        'Location': np.random.choice(['US', 'UK', 'CA', 'AU', 'IN'], 1000)
    })
    print("Created sample data for demonstration")

# Step 2: Data Understanding
print("\nStep 2: Data Understanding")
print("-" * 50)

# Basic information
print("\nBasic Information:")
print(data.info())

# Summary statistics
print("\nSummary Statistics:")
print(data.describe().round(2))

# Check for missing values
print("\nMissing Values:")
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found")

# Step 3: Data Preprocessing
print("\nStep 3: Data Preprocessing")
print("-" * 50)

# Create a copy of the data for preprocessing
processed_data = data.copy()

# Convert Timestamp to datetime if it exists
if 'Timestamp' in processed_data.columns:
    ##processed_data['Timestamp'] = pd.to_datetime(processed_data['Timestamp'])
    processed_data['Timestamp'] = pd.to_datetime(processed_data['Timestamp'], dayfirst=True, errors='coerce')

    # Create new time-based features
    processed_data['Hour'] = processed_data['Timestamp'].dt.hour
    processed_data['DayOfWeek'] = processed_data['Timestamp'].dt.dayofweek
    processed_data['Month'] = processed_data['Timestamp'].dt.month
    print("Created time-based features: Hour, DayOfWeek, Month")

# Handle missing values (if any)
if missing_values.sum() > 0:
    # For numerical columns: fill with median
    num_cols = processed_data.select_dtypes(include=np.number).columns
    processed_data[num_cols] = processed_data[num_cols].fillna(processed_data[num_cols].median())
    
    # For categorical columns: fill with most frequent value
    cat_cols = processed_data.select_dtypes(include=['object']).columns
    processed_data[cat_cols] = processed_data[cat_cols].fillna(processed_data[cat_cols].mode().iloc[0])
    print("Handled missing values")

# Encode categorical features
label_encoders = {}
categorical_columns = processed_data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    processed_data[col] = le.fit_transform(processed_data[col])
    label_encoders[col] = le
    print(f"Encoded categorical column: {col}")

# Step 4: Feature Engineering and Selection
print("\nStep 4: Feature Engineering and Selection")
print("-" * 50)

# Define features for anomaly detection
# Transaction amount and timing are typically good predictors
features = ['Amount']

# Add encoded categorical features if they exist
cat_features = ['Merchant', 'TransactionType', 'Location']
for feature in cat_features:
    if feature in processed_data.columns:
        features.append(feature)

# Add time-based features if they exist
time_features = ['Hour', 'DayOfWeek', 'Month']
for feature in time_features:
    if feature in processed_data.columns:
        features.append(feature)

print(f"Selected features: {features}")

# Step 5: Data Visualization
print("\nStep 5: Data Visualization")
print("-" * 50)

# Plot distribution of Amount
plt.figure(figsize=(10, 6))
sns.histplot(processed_data['Amount'], kde=True)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.savefig('amount_distribution.png')
print("Saved Amount distribution plot to 'amount_distribution.png'")

# If time features exist, create time-based visualizations
if 'Hour' in processed_data.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Hour', data=processed_data)
    plt.title('Transaction Count by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig('transactions_by_hour.png')
    print("Saved Transactions by hour plot to 'transactions_by_hour.png'")

if 'DayOfWeek' in processed_data.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='DayOfWeek', data=processed_data)
    plt.title('Transaction Count by Day of Week')
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig('transactions_by_day.png')
    print("Saved Transactions by day plot to 'transactions_by_day.png'")

# Plot amount by merchant (if exists)
if 'Merchant' in processed_data.columns and len(processed_data['Merchant'].unique()) < 20:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Merchant', y='Amount', data=processed_data)
    plt.title('Transaction Amount by Merchant')
    plt.xlabel('Merchant')
    plt.ylabel('Amount')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.savefig('amount_by_merchant.png')
    print("Saved Amount by merchant plot to 'amount_by_merchant.png'")

# Step 6: Anomaly Detection with One-Class SVM
print("\nStep 6: Anomaly Detection with One-Class SVM")
print("-" * 50)

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(processed_data[features])

# Apply One-Class SVM
print("Training One-Class SVM model...")
ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)  # nu is similar to contamination
ocsvm.fit(scaled_features)

# Predict anomalies
processed_data['OCSVM_Prediction'] = ocsvm.predict(scaled_features)
# Convert predictions: -1 for anomalies, 1 for normal points
processed_data['OCSVM_Anomaly'] = processed_data['OCSVM_Prediction'].apply(lambda x: 1 if x == -1 else 0)

# Identify potential fraudulent transactions
potential_fraud_ocsvm = processed_data[processed_data['OCSVM_Anomaly'] == 1]

print(f"One-Class SVM identified {len(potential_fraud_ocsvm)} potential fraudulent transactions")
print("\nSample of potential fraudulent transactions detected by One-Class SVM:")
print(potential_fraud_ocsvm.head(10))

# Calculate anomaly score
processed_data['OCSVM_Score'] = ocsvm.decision_function(scaled_features)
# Lower scores indicate higher likelihood of being anomalies


#Changs to try to improve statiscal to iqr
## Step 7: Simple Statistical Anomaly Detection (Enhance with domain knowledge)
#print("\nStep 7: Statistical Anomaly Detection")
#print("-" * 50)

## Use Z-score for Amount to detect outliers
#processed_data['Amount_ZScore'] = (processed_data['Amount'] - processed_data['Amount'].mean()) / processed_data['Amount'].std()
#processed_data['IQR_Anomaly'] = processed_data['Amount_ZScore'].apply(lambda x: 1 if abs(x) > 3 else 0)

## Transactions flagged as anomalous by statistical method
#statistical_anomalies = processed_data[processed_data['IQR_Anomaly'] == 1]
#print(f"Statistical method identified {len(statistical_anomalies)} potential fraudulent transactions")


# Additional Step: Feature Importance Analysis using SHAP
print("\nAdditional Step: Feature Importance Analysis using SHAP")
print("-" * 50)

import shap

# For computational efficiency, take a subset of scaled_features
sample_data = scaled_features[:200]

# Define a wrapper function for the decision_function of One-Class SVM.
# This function returns the anomaly score for each instance.
def svm_decision(x):
    return ocsvm.decision_function(x)

# Initialize the KernelExplainer with the model wrapper and a background dataset.
explainer = shap.KernelExplainer(svm_decision, sample_data)

# Calculate SHAP values on the sample (this may take some time)
shap_values = explainer.shap_values(sample_data, nsamples=100)

# Create a summary plot to visualize feature importance
shap.summary_plot(shap_values, sample_data, feature_names=features)


#############
# Step 7: IQR-based Anomaly Detection
print("\nStep 7: IQR-based Anomaly Detection")
print("-" * 50)

# Calculate Q1, Q3, and the Interquartile Range (IQR) for 'Amount'
Q1 = processed_data['Amount'].quantile(0.25)
Q3 = processed_data['Amount'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds for detecting outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Flag anomalies: set IQR_Anomaly to 1 if the 'Amount' is outside these bounds, else 0
processed_data['IQR_Anomaly'] = processed_data['Amount'].apply(
    lambda x: 1 if (x < lower_bound or x > upper_bound) else 0
)

# Transactions flagged as anomalous by the IQR method
iqr_anomalies = processed_data[processed_data['IQR_Anomaly'] == 1]
print(f"IQR-based method identified {len(iqr_anomalies)} potential fraudulent transactions")

'''
# Step 8: Comparison between methods
print("\nStep 8: Comparison between Methods")
print("-" * 50)

# Transactions flagged by both methods
both_methods_anomalies = processed_data[(processed_data['OCSVM_Anomaly'] == 1) & 
                                        (processed_data['IQR_Anomaly'] == 1)]
print(f"Number of transactions flagged by both methods: {len(both_methods_anomalies)}")

# Analyze characteristics of flagged transactions
if len(both_methods_anomalies) > 0:
    print("\nCharacteristics of transactions flagged by both methods:")
    for feature in features:
        if feature in ['Amount']:
            print(f"Average {feature}: {both_methods_anomalies[feature].mean():.2f} " +
                  f"(vs. {processed_data[feature].mean():.2f} overall)")
'''

# Step 8: Comparison between Methods
print("\nStep 8: Comparison between Methods")
print("-" * 50)

# Transactions flagged by both methods (One-Class SVM and IQR-based detection)
both_methods_anomalies = processed_data[(processed_data['OCSVM_Anomaly'] == 1) & 
                                        (processed_data['IQR_Anomaly'] == 1)]
print(f"Number of transactions flagged by both methods: {len(both_methods_anomalies)}")

# Analyze characteristics of flagged transactions for all numeric features
print("\nCharacteristics of transactions flagged by both methods:")
numeric_features = processed_data.select_dtypes(include=[np.number]).columns.tolist()

for feature in numeric_features:
    flagged_mean = both_methods_anomalies[feature].mean()
    overall_mean = processed_data[feature].mean()
    print(f"Average {feature}: {flagged_mean:.2f} (vs. {overall_mean:.2f} overall)")
###
# Step 9: Save processed data and results
print("\nStep 9: Save Results")
print("-" * 50)

# Save the processed data with anomaly flags
processed_data.to_csv('processed_data_with_anomalies.csv', index=False)
print("Saved processed data with anomaly detection results to 'processed_data_with_anomalies.csv'")

# Save potential fraud transactions
potential_fraud = processed_data[processed_data['OCSVM_Anomaly'] == 1]
potential_fraud.to_csv('potential_fraud_transactions.csv', index=False)
print("Saved potential fraud transactions to 'potential_fraud_transactions.csv'")

# Print summary
print("\nSummary:")
print("-" * 50)
print(f"Total transactions processed: {len(processed_data)}")
print(f"Potential fraudulent transactions (One-Class SVM): {len(potential_fraud_ocsvm)} ({len(potential_fraud_ocsvm)/len(processed_data)*100:.2f}%)")
#print(f"Potential fraudulent transactions (Statistical): {len(IQR_Anomaly)} ({len(IQR_Anomaly)/len(processed_data)*100:.2f}%)")
print(f"Potential fraudulent transactions (Statistical): {processed_data['IQR_Anomaly'].sum()} "
      f"({processed_data['IQR_Anomaly'].sum() / len(processed_data) * 100:.2f}%)")

print(f"Transactions flagged by both methods: {len(both_methods_anomalies)} ({len(both_methods_anomalies)/len(processed_data)*100:.2f}%)")

print("\nPhase 1 processing complete. Ready for Phase 2 with Isolation Forest.")
