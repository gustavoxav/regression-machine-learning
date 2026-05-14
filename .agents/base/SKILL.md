---
name: regression_salary_prediction_project
description: Helps with the development of a machine learning regression project focused on predicting Data Science salaries using preprocessing, linear regression, and polynomial regression techniques with different preprocessing configurations.
---

# Regression Salary Prediction Project

This skill supports the development of an academic machine learning project focused on salary prediction using the "Data Science Job Salaries" dataset from Kaggle.

The project objective is to predict the salary of professionals in the data science area based on variables such as experience level, employment type, job title, remote work ratio, company size, and geographic information.

The project must follow a complete machine learning workflow, including:
- exploratory data analysis;
- preprocessing;
- feature engineering;
- regression model training;
- comparison of preprocessing strategies;
- evaluation using regression metrics;
- analysis and discussion of results.

The initial scope of the project includes:
- Linear Regression;
- Polynomial Regression;
- preprocessing variations with and without dummy variables;
- preprocessing variations with and without data standardization.

Balancing techniques are not necessary because this is a regression problem.

## When to use this skill

- Use this when developing regression-based machine learning academic projects.
- Use this for projects involving salary prediction or numeric target prediction.
- Use this when comparing preprocessing strategies for regression models.
- Use this when generating preprocessing scripts, regression models, plots, reports, or article sections.
- Use this when analyzing regression metrics and comparing model performance.
- Use this for projects developed in Python using Anaconda and Spyder.

## Dataset Information

Dataset:
Data Science Job Salaries

Target variable:
- salary_in_usd

Available attributes:
- work_year
- experience_level
- employment_type
- job_title
- salary
- salary_currency
- employee_residence
- remote_ratio
- company_location
- company_size

## Recommended Project Workflow

### 1. Data Loading

- Load the dataset using pandas.
- Use `.describe()` and `.info()` for initial inspection.
- Verify missing values using:
  ```python
  pd.isnull(base).any()


### 2. Data Cleaning

* Remove duplicated rows if necessary.
* Remove columns that may cause leakage:

  * salary
  * salary_currency
* Keep:

  * salary_in_usd as target variable.

### 3. Feature Selection

Recommended predictors:

```python
cols_previsores = [
    'work_year',
    'experience_level',
    'employment_type',
    'job_title',
    'employee_residence',
    'remote_ratio',
    'company_location',
    'company_size'
]
```

Target:

```python
cols_objetivo = ['salary_in_usd']
```

### 4. Categorical Encoding

Generate different preprocessing variations:

#### Option 1 — Label Encoding

Use LabelEncoder for categorical attributes.

Recommended for:

* Linear Regression baseline tests;
* compact datasets.

#### Option 2 — One-Hot Encoding (Dummy Variables)

Use:

```python
pd.get_dummies()
```

Recommended for:

* avoiding artificial ordinal relationships;
* improving regression performance.

### 5. Train/Test Split

Use:

```python
train_test_split(
    previsores,
    objetivo,
    test_size=0.25,
    random_state=0
)
```

### 6. Standardization

Test models:

* with StandardScaler;
* without StandardScaler.

Use:

```python
from sklearn.preprocessing import StandardScaler
```

Apply scaler only to predictors.

### 7. Regression Models

Initial models:

#### Linear Regression

```python
from sklearn.linear_model import LinearRegression
```

#### Polynomial Regression

Use:

```python
from sklearn.preprocessing import PolynomialFeatures
```

Test different polynomial degrees:

* degree=2
* degree=3

Avoid very high degrees to reduce overfitting.

## Suggested Dataset Variations

Generate preprocessing combinations:

* sem_dummy_sem_std
* sem_dummy_com_std
* com_dummy_sem_std
* com_dummy_com_std

These variations allow comparative analysis of:

* categorical encoding impact;
* normalization impact;
* model sensitivity to feature scaling.

## Recommended Metrics

Use multiple regression metrics:

### MAE

Mean Absolute Error.

Represents average prediction error.

### MSE

Mean Squared Error.

Penalizes larger errors.

### RMSE

Root Mean Squared Error.

More interpretable than MSE because it uses original scale.

### R² Score

Coefficient of determination.

Measures how much variance the model explains.

## Expected Analysis

The project should analyze:

* effect of preprocessing techniques;
* effect of polynomial degree;
* overfitting behavior;
* comparison between train and test metrics;
* feature importance or coefficient interpretation;
* influence of remote work, experience level, and company size on salary prediction.

## Suggested Visualizations

Generate:

* correlation matrix;
* regression plots;
* predicted vs actual values;
* residual plots;
* metric comparison bar charts;
* polynomial fitting comparison graphs.

## Important Academic Discussion Points

* Impact of experience level on salary.
* Relationship between remote work and salaries.
* Geographic influence on compensation.
* Effect of company size.
* Model generalization capability.
* Trade-off between model complexity and overfitting.

## Code Style Recommendations

* Use clear section divisions with comments.
* Separate preprocessing, training, and evaluation stages.
* Save plots and results to folders.
* Use reproducible seeds (`random_state`).
* Maintain compatibility with Spyder and Anaconda environments.

## Expected Final Deliverables

The final project should contain:

* preprocessing scripts;
* regression model scripts;
* generated datasets;
* charts and figures;
* confusion-free folder organization;
* LaTeX article;
* presentation slides;
* analysis of metrics and conclusions.
