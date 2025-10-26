# Medical Equipment Transport Cost Prediction 🏥📦

> **Team: Unsupervised Learners**  
> R. Sreenivasa Raju (IMT2023122) | U. Trivedh Venkata Sai (IMT2023002)

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://www.kaggle.com)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

---

## 📋 Project Overview

This project tackles a **regression challenge** to predict transport costs for medical equipment deliveries. The solution leverages advanced feature engineering, robust preprocessing pipelines, and systematic model evaluation to achieve competitive performance on the Kaggle platform.

### 🎯 Business Problem

Medical equipment logistics providers need accurate cost predictions to:
- Provide competitive pricing quotes
- Optimize logistics planning and resource allocation
- Manage risk in high-cost delivery scenarios (fragile, urgent, cross-border)
- Negotiate fair contracts with shipping partners

---

## 📊 Dataset

**Source:** Kaggle Medical Equipment Transport Cost Prediction Challenge

| Dataset | Samples | Features |
|---------|---------|----------|
| Training | 5,000 | 20 |
| Test | 500 | 19 |

**Target Variable:** `Transport_Cost` (continuous, USD)  
**Evaluation Metric:** Root Mean Squared Error (RMSE)

### Key Features:
- **Equipment:** Height, Width, Weight, Value, Type, Fragility
- **Logistics:** Base Fee, Transport Method, Supplier Reliability
- **Delivery:** Cross-Border, Urgent, Installation Service, Rural Hospital
- **Temporal:** Order Date, Delivery Date

---

## 🔧 Technical Approach

### 1. **Data Preprocessing Pipeline**

Built a modular `ColumnTransformer` with 6 parallel preprocessing streams:

```python
Pipeline:
├── Numeric (with missing values) → Median Imputation → RobustScaler
├── Categorical (with missing values) → Mode Imputation → OneHotEncoder
├── Date features → Temporal Extraction → Cyclical Encoding
├── Complete numeric → RobustScaler
├── Complete categorical → OneHotEncoder
└── Engineered features → Median Imputation → RobustScaler
```

**Key Strategies:**
- **Missing Values:** Median for numeric, "Unknown" category for categorical
- **Encoding:** One-hot for categorical, cyclical (sin/cos) for temporal
- **Scaling:** RobustScaler for outlier resistance
- **Target Transform:** PowerTransformer (Yeo-Johnson) for skewed cost distribution

### 2. **Feature Engineering**

Created **6 custom features** via `EquipmentFeatureAdder` transformer:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `ValuePerKg` | Value / Weight | High-value density → enhanced security |
| `BaseCostPerKg` | Base_Fee / Weight | Normalized pricing across weights |
| `CrossBorderUrgent` | CrossBorder × Urgent | Multiplicative cost for urgent international |
| `FragileUrgent` | Fragile × Urgent | Premium for time-sensitive delicate items |
| `RuralCrossBorder` | Rural × CrossBorder | Compounded logistics difficulty |
| `ComplexShipping` | Sum of flags | Aggregate delivery complexity |

**Temporal Features:**
- Delivery duration (days)
- Cyclical encoding (day-of-week, month) using sin/cos transformations
- Weekend order/delivery indicators

### 3. **Model Benchmarking**

Evaluated **7 algorithms** with GridSearchCV (3-Fold CV):

| Model | R² Score | RMSE (USD) | Training Time |
|-------|----------|------------|---------------|
| **ElasticNet** ✅ | **0.294** | **39,576** | <1s |
| RandomForest | 0.291 | 39,652 | ~30s |
| BayesianRidge | 0.274 | 40,138 | <1s |
| Ridge | 0.261 | 40,493 | <1s |
| Lasso | 0.260 | 40,530 | <1s |
| AdaBoost | 0.171 | 42,890 | ~20s |
| XGBoost ❌ | -0.200 | 51,586 | ~15s |

**Selected Model:** ElasticNet (optimal balance of accuracy, speed, interpretability)

---

## 🚀 Key Results

### Performance Metrics

- **Validation RMSE:** $39,576
- **Validation R²:** 0.294 (explains ~30% of variance)
- **Prediction Error:** 10-15% for typical mid-range shipments

### Top Cost Drivers

1. **Base Transport Fee** (strongest predictor)
2. **Cross-Border Shipping** (~50-100% premium)
3. **Urgent Shipping** (~25-30% premium)
4. **Equipment Value** (~20-30% increase for high-value)
5. **Rural Hospitals** (~30-40% premium)

### Feature Engineering Impact

- **Baseline (raw features):** RMSE ~$48,000
- **With engineered features:** RMSE $39,576
- **Improvement:** ~17% reduction in RMSE ✨

---

## 💡 Key Insights & Lessons

### What Worked ✅

1. **Custom Feature Engineering:** Domain-informed features > algorithm sophistication
2. **Pipeline Architecture:** Prevented data leakage, ensured reproducibility
3. **Cyclical Encoding:** Preserved circular relationships (weekdays, months)
4. **Target Transformation:** Improved linear model performance

### What Didn't Work ❌

1. **XGBoost:** Negative R² despite extensive tuning
   - **Lesson:** Complex models don't always win; 5,000 samples favor simpler models
2. **Deep Learning:** Insufficient data (5,000 samples)
   - **Lesson:** Dataset size constraints matter
3. **All Pairwise Interactions:** Combinatorial explosion
   - **Lesson:** Domain knowledge beats brute force

---

## 📁 Repository Structure

```
├── ultimate_ml_assignment.ipynb   # Main notebook with complete pipeline
├── train.csv                       # Training data
├── test.csv                        # Test data
├── Medical-Transport-Report.pdf   # Detailed technical report
└── README.md                       # This file
```

---

## 🛠️ Installation & Usage

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost
```

### Quick Start

```python
# Load and run the notebook
jupyter notebook ultimate_ml_assignment.ipynb
```

### Pipeline Usage

```python
from sklearn.pipeline import Pipeline

# Complete pipeline structure
pipeline = Pipeline([
    ('feature_adder', EquipmentFeatureAdder()),
    ('preprocessor', ColumnTransformer(...)),
    ('regressor', TransformedTargetRegressor(
        regressor=ElasticNetCV(...),
        transformer=PowerTransformer(...)
    ))
])

# Fit and predict
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

---

## 📈 Future Improvements

### Short-Term
- Collect more training data (target: 15,000+ samples)
- Add geographic features (haversine distance, route complexity)
- Incorporate real-time factors (fuel prices, weather)

### Medium-Term
- Model ensembling (stack ElasticNet + RandomForest)
- Time series modeling for seasonal trends
- Uncertainty quantification (prediction intervals)

### Long-Term
- Deep learning with larger dataset
- Route optimization integration
- Dynamic pricing with real-time adjustments

---

## 👥 Team Contributions

**R. Sreenivasa Raju (IMT2023122)**
- Feature engineering design
- Model benchmarking and hyperparameter tuning
- Final evaluation and validation

**U. Trivedh Venkata Sai (IMT2023002)**
- Data preprocessing pipeline
- Pipeline automation and integration
- Results analysis and documentation

---

## 📝 License

This project is part of an academic ML assignment. For educational purposes only.

---

## 🙏 Acknowledgments

- Kaggle for hosting the competition
- scikit-learn community for excellent documentation
- Healthcare logistics domain experts for insights

---

## 📞 Contact

For questions or collaboration:
- R. Sreenivasa Raju: [IMT2023122]
- U. Trivedh Venkata Sai: [IMT2023002]

---

**⭐ If you find this project helpful, please star the repository!**