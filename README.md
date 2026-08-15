# 📊 Telco Customer Churn Prediction

> **An end-to-end machine learning project for predicting customer churn using the IBM Telco Customer Churn dataset — from exploratory analysis and statistical testing to model tuning, interpretability, calibration, error analysis, and production-ready pipeline persistence.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Overview

Customer churn is one of the most important problems for subscription-based businesses.

The goal of this project is to build a machine learning system capable of identifying customers who are likely to churn, allowing businesses to proactively target high-risk customers with retention strategies.

Rather than treating this as a simple model-training exercise, this project implements a **complete supervised learning workflow**, including:

* Business problem framing
* Exploratory Data Analysis
* Data cleaning and preprocessing
* Statistical hypothesis testing
* Outlier analysis
* Feature engineering
* Multicollinearity analysis using VIF
* Leakage-safe preprocessing pipelines
* Class-imbalance handling
* Multiple model families
* Stratified cross-validation
* Hyperparameter optimization
* Bias/variance analysis
* Probability calibration
* Model interpretability
* Error analysis
* Model persistence
* Raw-input prediction
* Experiment tracking

The final model is selected based on both **machine learning performance and business relevance**, rather than accuracy alone.

---

# 🎯 Problem Statement

The dataset contains information about customers of a telecommunications company and whether they eventually churned.

### Objective

> **Predict whether a customer is likely to churn based on their demographic information, account characteristics, services, contract details, and billing information.**

### Problem Type

**Binary Classification**

| Value | Meaning          |
| ----- | ---------------- |
| `Yes` | Customer churned |
| `No`  | Customer stayed  |

### Target Variable

`Churn`

### Dataset

* **7,043 customers**
* **21 original features**
* Binary classification target
* Combination of numerical and categorical variables

---

# 💼 Business Perspective

A churn prediction model is useful only if its predictions support better business decisions.

This project therefore considers the different costs of prediction errors.

| Prediction    | Actual Outcome | Interpretation                            |
| ------------- | -------------- | ----------------------------------------- |
| Predict Churn | Churn          | ✅ Correctly identifies a customer at risk |
| Predict Stay  | Stay           | ✅ Correct prediction                      |
| Predict Churn | Stay           | ⚠️ Unnecessary retention effort           |
| Predict Stay  | Churn          | ❌ Missed churn opportunity                |

A **false negative** can be particularly expensive because the business may completely miss an opportunity to retain a customer.

For this reason, the project places greater emphasis on:

* **Recall**
* **F1-score**
* ROC-AUC
* Precision-Recall performance

rather than relying on accuracy alone.

---

# 🧠 Machine Learning Workflow

The project follows a complete ML lifecycle:

```text
Business Problem
       ↓
Exploratory Data Analysis
       ↓
Data Cleaning
       ↓
Statistical Testing
       ↓
Outlier Analysis
       ↓
Feature Engineering
       ↓
Multicollinearity Analysis
       ↓
Train / Test Split
       ↓
Leakage-Safe Preprocessing
       ↓
Baseline Model
       ↓
Class Imbalance Handling
       ↓
Model Comparison
       ↓
5-Fold Stratified Cross-Validation
       ↓
Hyperparameter Tuning
       ↓
Bias / Variance Analysis
       ↓
Final Test Evaluation
       ↓
Probability Calibration
       ↓
Feature Importance
       ↓
Error Analysis
       ↓
Final Model Selection
       ↓
Model Persistence
       ↓
Prediction on Raw Input
```

---

# 🔎 Exploratory Data Analysis

The notebook performs a detailed exploration of the dataset before modeling.

The analysis includes:

* Target distribution
* Numerical feature distributions
* Categorical feature distributions
* Correlation analysis
* Churn rate comparisons
* Boxplots
* Missing-value analysis
* Relationships between customer characteristics and churn

Special attention is given to `TotalCharges`, which requires conversion to numeric form because the original dataset contains blank values for some customers.

---

# 📐 Statistical Analysis

Categorical variables are evaluated using the **Chi-Square Test of Independence** to determine whether there is a statistically significant relationship with customer churn.

The analysis identifies important relationships involving variables such as:

* `Contract`
* `InternetService`
* `OnlineSecurity`
* `TechSupport`
* `PaymentMethod`

This statistical analysis is used alongside model-based feature importance rather than being treated as a replacement for predictive modeling.

---

# 🛠️ Feature Engineering

The project explores additional features including:

### Average Monthly Charge

A derived billing-related feature used to investigate customer spending patterns.

### Tenure Groups

Customers are grouped into tenure categories to investigate how customer lifetime relates to churn behavior.

The notebook also performs **Variance Inflation Factor (VIF)** analysis to investigate multicollinearity.

An important modeling decision is made after this analysis: highly redundant engineered variables are not unnecessarily added to the final modeling pipeline.

This keeps the final feature space cleaner while preserving predictive information.

---

# ⚠️ Class Imbalance

Customer churn is a minority-class problem in this dataset.

Approximately:

* **73%** → Customers who stayed
* **27%** → Customers who churned

This makes accuracy alone a poor metric.

For example, a model predicting "No Churn" for almost everyone could still achieve high accuracy while performing poorly at actually detecting churners.

Therefore, the project experiments with two approaches:

### `class_weight="balanced"`

Automatically increases the importance of the minority class during model training.

### SMOTE

Synthetic Minority Over-sampling Technique is used to generate additional minority-class training examples.

SMOTE is implemented inside the appropriate pipeline so that synthetic samples are generated only from training data during cross-validation.

This prevents validation/test leakage.

---

# 🤖 Models Compared

Six model approaches are evaluated:

| Model                    | Approach                                     |
| ------------------------ | -------------------------------------------- |
| Logistic Regression — L1 | Linear classification with L1 regularization |
| Logistic Regression — L2 | Linear classification with L2 regularization |
| K-Nearest Neighbors      | Distance-based classification                |
| Decision Tree            | Recursive rule-based splitting               |
| Random Forest            | Ensemble of decision trees                   |
| Gradient Boosting        | Sequential boosting of weak learners         |

Every model is evaluated using the same general preprocessing and validation framework to make the comparison meaningful.

---

# 🔬 Cross-Validation

Model comparison uses:

**5-Fold Stratified Cross-Validation**

The primary optimization metric is:

**F1-score**

For every experiment, the notebook tracks:

* Mean CV F1
* CV F1 standard deviation
* Test performance
* Model parameters
* Experiment notes

Stratification ensures that the churn/stay distribution remains approximately consistent across validation folds.

---

# 🎛️ Hyperparameter Tuning

The strongest candidate models are further optimized using:

**GridSearchCV**

Hyperparameters are tuned for:

* Logistic Regression
* Random Forest
* Gradient Boosting

The same F1-based evaluation strategy is maintained during tuning to ensure consistency with the model-selection objective.

---

# 📊 Final Results

The final experiments produced the following results:

| Model                        |     CV F1 |   Test F1 | Test ROC-AUC | Test Recall |
| ---------------------------- | --------: | --------: | -----------: | ----------: |
| 🏆 **Tuned Random Forest**   | **0.633** | **0.632** |    **0.841** |   **0.751** |
| Tuned Logistic Regression    |     0.630 |     0.618 |        0.841 |   **0.786** |
| Tuned Gradient Boosting      |     0.590 |     0.590 |    **0.843** |       0.524 |
| Logistic Regression Baseline |         — |     0.604 |        0.842 |       0.559 |

---

# 🏆 Final Model

## Tuned Random Forest

The **Tuned Random Forest** was selected as the final model.

### Why Random Forest?

It achieved:

* 🥇 Best cross-validated F1-score
* 🥇 Best test-set F1-score
* Strong recall
* ROC-AUC comparable to the other top models
* Good probability calibration
* Strong feature interpretability
* Consistent performance between cross-validation and the held-out test set

### Final Performance

**Test F1:** `0.632`

**Test ROC-AUC:** `0.841`

**Test Recall:** `0.751`

The model therefore provides a strong overall balance between identifying churners and avoiding unnecessary retention actions.

Although Gradient Boosting achieved a marginally higher ROC-AUC, its recall was considerably lower. Under the business objective of minimizing missed churners, Random Forest is the more appropriate final choice.

---

# 🔥 Key Churn Drivers

Multiple independent analyses point toward several important churn signals.

The strongest predictors include:

### 📅 Month-to-Month Contracts

Customers without long-term contracts show substantially higher churn risk.

### ⏳ Low Tenure

Newer customers are significantly more likely to churn than long-term customers.

### 🌐 Fiber-Optic Internet

Fiber-optic customers show an elevated churn pattern in this dataset.

### 💰 Higher Monthly Charges

Higher monthly billing is associated with increased churn risk.

### 🔐 Lack of Online Security

Customers without online security services show higher churn tendencies.

### 🛠️ Lack of Technical Support

Customers without technical support are also more likely to churn.

These patterns are supported across multiple approaches including:

* Exploratory analysis
* Chi-square testing
* Random Forest feature importance
* Permutation importance

---

# 🧪 Bias & Variance Analysis

The notebook does not simply compare model scores.

It also investigates **why models perform differently**.

Learning curves and controlled Decision Tree experiments are used to demonstrate:

### High Variance / Overfitting

A highly complex model can perform extremely well on training data while generalizing poorly.

### High Bias / Underfitting

A heavily restricted model may be too simple to capture the underlying relationships in the data.

This analysis helps explain the trade-off between:

**Model Complexity ↔ Generalization**

---

# 📈 Model Evaluation

The final model is evaluated using multiple complementary metrics:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Log Loss
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve

Using multiple metrics provides a more complete picture of model performance than accuracy alone.

---

# 🎯 Probability Calibration

Classification models do not only produce class labels.

They can also produce probabilities such as:

> "This customer has an estimated 78% probability of churning."

For that reason, the project evaluates **probability calibration**.

A well-calibrated model is useful when predictions are used to:

* Rank customers by risk
* Prioritize retention campaigns
* Define different intervention levels
* Support business decision-making

---

# 🔍 Model Interpretability

The final Random Forest is analyzed using two complementary feature-importance methods.

### Built-in Feature Importance

Uses the importance values calculated by the Random Forest.

### Permutation Importance

Features are randomly shuffled and the resulting change in model performance is measured.

Using both approaches provides a stronger understanding of which variables contribute to predictions.

---

# 🧩 Error Analysis

The project also investigates incorrectly classified customers, especially difficult or high-confidence errors.

This helps identify:

* Ambiguous customer profiles
* Difficult customer segments
* Potentially missing predictive variables
* Limitations of the current model

Error analysis is treated as an important part of the modeling process rather than simply reporting a final score.

---

# 🔒 Leakage-Safe ML Pipeline

One of the key design goals of this project is preventing **data leakage**.

The preprocessing workflow uses:

* `ColumnTransformer`
* `Pipeline`
* Numerical imputation
* Categorical imputation
* Feature scaling
* One-hot encoding

The preprocessing steps are fitted only on training data.

This ensures that information from the validation or test set is not accidentally used during model training.

This is particularly important when using:

* Cross-validation
* SMOTE
* Hyperparameter tuning
* Feature transformations

---

# 💾 Model Persistence

The final model pipeline is persisted using **Joblib**.

The saved pipeline contains both:

**Preprocessing + Model**

This means a new raw customer record can be passed through the same preprocessing and prediction workflow without manually repeating:

* Missing-value handling
* Scaling
* Categorical encoding
* Feature transformation

The notebook also includes a raw-input `predict_new()` workflow demonstrating that the final pipeline can accept customer data in its original form.

---

# 🧾 Experiment Tracking

The notebook maintains an experiment log containing information such as:

* Model name
* Hyperparameters
* Cross-validation F1
* Test metrics
* Experiment notes

This makes the modeling process reproducible and provides a lightweight experiment-tracking system.

---

# 📁 Repository Structure

```text
telco-customer-churn-prediction/
│
├── Telco_Customer_Churn.ipynb
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── README.md
│
└── telco_churn_pipeline.joblib
```

> `telco_churn_pipeline.joblib` is generated when the model-persistence section of the notebook is executed.

---

# 🧰 Tech Stack

| Technology       | Purpose                         |
| ---------------- | ------------------------------- |
| Python           | Core programming language       |
| Pandas           | Data manipulation               |
| NumPy            | Numerical computation           |
| Matplotlib       | Visualization                   |
| Seaborn          | Statistical visualization       |
| Scikit-learn     | Machine learning                |
| Imbalanced-learn | SMOTE and imbalance handling    |
| Statsmodels      | VIF / statistical analysis      |
| Joblib           | Model persistence               |
| Jupyter Notebook | Development and experimentation |

---

# ▶️ Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/Harshad2321/telco-customer-churn-prediction.git
cd telco-customer-churn-prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn statsmodels joblib jupyter
```

### 3. Launch Jupyter

```bash
jupyter notebook
```

### 4. Open

```text
Telco_Customer_Churn.ipynb
```

Run the notebook from top to bottom to reproduce the complete analysis.

---

# 📚 Dataset

This project uses the **IBM Telco Customer Churn Dataset**.

**Dataset characteristics:**

* 7,043 customer records
* 21 columns
* Demographic information
* Account information
* Service information
* Billing information
* Churn label

### Source

[Kaggle — IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

# 🔮 Future Improvements

The current project provides a strong end-to-end baseline, but several extensions could make it more production-oriented.

### Potential improvements

* SHAP-based explanations
* Customer-level prediction explanations
* Business-cost-based threshold optimization
* Churn-risk ranking
* Automated model monitoring
* Data drift detection
* Model drift detection
* MLflow experiment tracking
* REST API deployment
* Batch churn prediction
* Retention campaign integration
* Automated retraining pipeline

---

# 🎓 What This Project Demonstrates

This project demonstrates practical understanding of the complete supervised machine learning workflow rather than only model training.

### Machine Learning

* Binary classification
* Logistic Regression
* KNN
* Decision Trees
* Random Forest
* Gradient Boosting
* Ensemble learning
* Hyperparameter tuning

### Data Science

* EDA
* Statistical testing
* Feature engineering
* Outlier analysis
* Multicollinearity
* Data preprocessing

### Model Evaluation

* Cross-validation
* Precision
* Recall
* F1-score
* ROC-AUC
* Log Loss
* Calibration
* Confusion matrices
* ROC curves
* Precision-Recall curves

### ML Engineering

* Leakage prevention
* Pipelines
* ColumnTransformer
* SMOTE
* Model persistence
* Raw-input prediction
* Experiment logging

### Business Thinking

* Cost of false negatives
* Churn-risk prioritization
* Retention strategy
* Model selection based on business objectives

---

# 💡 Key Takeaway

The main objective of this project was not simply to find a model with the highest accuracy.

Instead, the project demonstrates how to move from:

> **Business Problem → Data → Analysis → Statistical Validation → Modeling → Tuning → Evaluation → Interpretation → Error Analysis → Deployment Readiness**

The final **Tuned Random Forest** achieved a **0.632 test F1-score**, **0.841 ROC-AUC**, and **0.751 recall**, making it the strongest overall model under the project's churn-detection objective.

The project therefore serves as a complete example of a **leakage-safe, evaluation-driven supervised machine learning workflow**.

---

## 👨‍💻 Author

**Harshad Agrawal**

B.Tech — Artificial Intelligence & Machine Learning

Symbiosis Institute of Technology, Pune

---

## 📄 License

This project is released under the **MIT License**.

Feel free to use it for learning, experimentation, research, and portfolio purposes.
