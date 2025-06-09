# ML-Fraud-Detection-Unlabled

# ML Fraud Detection with Unlabeled Dataset

A machine learning project for detecting fraudulent transactions using unsupervised learning techniques on an unlabeled dataset.

## Team Members

- **Ashish D Fugare** (Team Leader)  
  Roll: 170101023 | ashis170101023@iitg.ac.in | Group: Sinhagad
  
- **Priya Sharma**  
  Roll: 22001081 | priya.sharma@iitg.ac.in | Group: Sinhagad

## Problem Statement

Perform fraud transaction detection on a dataset that lacks labels identifying which transactions are fraudulent or legitimate. This presents a classic unsupervised learning challenge where we must identify anomalous patterns without ground truth data.

## Dataset Overview

- **Total transactions**: 217,441
- **Features**: 7 (Timestamp, TransactionID, AccountID, Amount, TransactionType, Merchant, Location)
- **Missing entries**: 481 (removed from analysis)
- **Data split**: Training (70%), Test (20%), Validation (10%)
- **Preprocessing**: Removed TransactionID as it doesn't contribute to fraud detection

## Feature Engineering

We engineered 15+ additional features to better capture fraudulent behavior patterns:

### Account-Based Features
- **Avg Amount**: Average transaction amount per account (baseline spending behavior)
- **Std Amount**: Standard deviation of amounts per account (spending variability)
- **Min/Max Amount**: Transaction boundaries for each account
- **Tx Count**: Number of transactions per account (activity level)
- **Amount Ratio**: Current transaction / average amount (unusual spending detection)

### Time-Based Features
- **Hour/Day/DayOfWeek/Month**: Temporal transaction patterns
- **Weekend**: Binary flag for weekend transactions
- **BusinessHours**: Binary flag for 9am-5pm weekday transactions
- **Time Since Last Tx**: Hours since previous transaction (frequency analysis)
- **Rapid Succession**: Flag for transactions < 1 hour apart

### Location-Based Features
- **Common Location**: Most frequent transaction location per account
- **Location Change**: Flag for transactions in unusual locations

## Methodology

### Initial Approach: Clustering (DBSCAN)
- **Result**: Poor performance with only 15 flagged transactions
- **Silhouette Score**: 0.084 (indicating poor clustering)
- **Issue**: Fraudulent transactions don't cluster together; they're scattered throughout feature space with subtle differences

### Final Approach: Anomaly Detection

We implemented two complementary models:

#### 1. Isolation Forest
- **Principle**: Anomalies require fewer splits to isolate
- **Strength**: Excels at detecting amount-based anomalies and day-of-week patterns
- **Example**: $50,000 Sunday transactions from accounts averaging $500 on weekdays

#### 2. Local Outlier Factor (LOF)
- **Principle**: Identifies points in low-density regions relative to neighbors
- **Strength**: Better at timing anomalies and sequence irregularities
- **Example**: 3 AM transactions from accounts normally active 9 AM-5 PM

### Threshold Selection
- **Chosen threshold**: 5% (industry standard for fraud rates: 1-10%)
- **Rationale**: Balance between catching fraud and avoiding false positives
- **Consistency**: Both models flagged ~5% across validation and test sets

## Results

```
Model Performance (5% contamination rate):
- Isolation Forest threshold: 0.000223
- LOF threshold: -0.002165
- Isolation Forest anomalies: 2,175 (5.00%)
- LOF anomalies: 2,175 (5.00%)
- Ensemble anomalies: 3,979 (9.15%)
- Model overlap: Only 387 transactions (indicating complementary detection)
```

### Validation Results
- **Validation Set**: IF (5.03%), LOF (5.59%)
- **Test Set**: IF (5.22%), LOF (5.07%)

## Feature Importance Analysis

### Most Significant Features

1. **Min Amount** (Z ≈ 190): Fraudsters often conduct exclusively large transactions
2. **Time Since Last Tx** (Z ≈ 8): Extended inactivity followed by sudden activity
3. **Location Change** (0.54 total variation): Transactions across different cities
4. **Tx Count**: Negative correlation (fraudsters minimize transaction counts)

### Feature Subset Performance
- **Transaction Type alone**: 33% model agreement
- **Merchant features**: 10% agreement  
- **Time features**: 9.4% agreement
- **All features combined**: 7.2% agreement

*Note: Specific transaction types show strong fraud signals, while combining all features introduces noise.*

## Key Learnings

### Technical Insights
1. **Data Handling**: Excel auto-formatting caused timestamp parsing errors in Python
2. **Feature Scaling**: Critical for performance - without scaling, models focused too heavily on amount features
3. **Manual Implementation**: Custom LOF matched scikit-learn exactly; Isolation Forest showed slight differences due to randomness
4. **Performance**: Manual implementations were significantly slower than optimized libraries

### Domain Insights
1. **Feature Engineering**: Derived features (Time Since Last Tx, Location Change) more important than raw data
2. **Model Complementarity**: Different algorithms capture different fraud patterns
3. **Threshold Selection**: Industry knowledge crucial for unsupervised learning

## Implementation Notes

- **Environment**: Python with scikit-learn, pandas, numpy
- **Scaling**: StandardScaler applied to all numerical features
- **Validation**: Cross-validation across train/validation/test splits
- **Evaluation**: Statistical tests and correlation analysis for feature importance

## Repository Structure

```
ML-Fraud-Detection-Unlabeled/
├── README.md
├── data/
│   └── transactions.csv
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── evaluation.py
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── model_comparison.ipynb
└── results/
    ├── anomaly_scores.csv
    └── feature_importance.csv
```

## Usage

```python
# Basic usage example
from src.models import FraudDetector

# Initialize detector
detector = FraudDetector(contamination=0.05)

# Fit models
detector.fit(X_train)

# Predict anomalies
anomalies = detector.predict(X_test)
```

## Future Work

- Experiment with other anomaly detection algorithms (One-Class SVM, Autoencoders)
- Implement ensemble methods with weighted voting
- Develop real-time fraud detection pipeline
- Validate approach with labeled datasets when available

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- IIT Guwahati for providing the computational resources
- Industry research on fraud detection patterns and methodologies
