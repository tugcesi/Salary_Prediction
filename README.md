# Salary Prediction with Machine Learning

A machine learning project that predicts employee salaries based on various features using multiple regression algorithms.

## Dataset

The `Salary_Data.csv` dataset contains employment information with the following features:

| Feature | Description |
|---------|-------------|
| Age | Employee age |
| Gender | Male / Female |
| Education Level | High School, Bachelor's, Master's, PhD |
| Job Title | Employee job role |
| Years of Experience | Total years of work experience |
| Salary | Target variable — annual salary in USD |

## Models & Performance

Nine regression algorithms were trained and evaluated. Results are sorted by R² Score (higher is better):

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| **XGBRegressor** | **0.954072** | **11,297.38** | **6,156.61** |
| Extra Tree | 0.948234 | 11,993.86 | 5,252.84 |
| Decision Tree | 0.947939 | 12,027.96 | 5,269.94 |
| KNeighborsRegressor | 0.941537 | 12,746.16 | 6,265.76 |
| Gradient Boosting | 0.897086 | 16,911.19 | 12,241.72 |
| Lasso | 0.761792 | 25,728.58 | 19,804.42 |
| Linear Regression | 0.761772 | 25,729.64 | 19,811.69 |
| Ridge | 0.761743 | 25,731.22 | 19,803.13 |
| ElasticNet | 0.714184 | 28,182.59 | 22,498.07 |

> **Best Model:** XGBRegressor with R² = 0.954072 (explains 95.4% of salary variance), RMSE = $11,297.38, MAE = $6,156.61

## Project Structure

```
├── SalaryPredictionwithMachineLearning.ipynb   # Main notebook with all models
├── Salary_Data.csv                              # Dataset
├── README.md
└── .gitignore
```

## Notebook Contents

1. **Import Libraries** — numpy, pandas, scikit-learn, xgboost, matplotlib, seaborn
2. **Load Dataset** — Load and inspect `Salary_Data.csv`
3. **Exploratory Data Analysis** — Distribution plots, correlation heatmap
4. **Data Preprocessing** — Label encoding, ordinal mapping, standard scaling
5. **Feature Selection & Train-Test Split** — 80/20 split with `random_state=42`
6. **Model Training & Evaluation** — All 9 regression models
7. **Model Comparison** — Bar charts for R² and RMSE
8. **Best Model Analysis** — Actual vs. predicted plot, feature importance
9. **Conclusion** — Summary table and key findings

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
```

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/tugcesi/Salary_Prediction.git
   cd Salary_Prediction
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost
   ```

3. Open the notebook:
   ```bash
   jupyter notebook SalaryPredictionwithMachineLearning.ipynb
   ```

4. Run all cells to train and evaluate models.

## Key Findings

- Tree-based ensemble models (XGBoost, Extra Trees) significantly outperform linear models for salary prediction.
- **Years of Experience** and **Job Title** are the most important features.
- Linear models (Linear Regression, Ridge, Lasso, ElasticNet) have lower performance because the salary–feature relationship is non-linear.
