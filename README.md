# Salary_Prediction
Salary Prediction with Machine Learning
# Salary Prediction with Machine Learning

## 📊 Project Overview
This project aims to predict employee salaries using various machine learning algorithms. The dataset contains employee information including age, gender, education level, job title, and years of experience.

## 🏆 Best Performing Model
**XGBRegressor** achieves the highest performance with:
- **R² Score: 0.954072** (explains 95.4% of salary variance)
- **RMSE: $11,297.38**
- **MAE: $6,156.61**

## 📈 Model Performance Comparison

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| **XGBRegressor** | **0.954072** | **11297.38** | **6156.61** |
| Extra Tree | 0.948234 | 11993.86 | 5252.84 |
| Decision Tree | 0.947939 | 12027.96 | 5269.94 |
| KNeighborsRegressor | 0.941537 | 12746.16 | 6265.76 |
| Gradient Boosting | 0.897086 | 16911.19 | 12241.72 |
| Lasso | 0.761792 | 25728.58 | 19804.42 |
| Linear Regression | 0.761772 | 25729.64 | 19811.69 |
| Ridge | 0.761743 | 25731.22 | 19803.13 |
| ElasticNet | 0.714184 | 28182.59 | 22498.07 |

## 📋 Dataset Features
- **Age**: Employee age (21-62 years)
- **Gender**: Male, Female, Other
- **Education Level**: High School, Bachelor's, Master's, PhD
- **Job Title**: Position/Role
- **Years of Experience**: 0-34 years
- **Salary**: Target variable ($350 - $250,000)

## 📊 Dataset Statistics
- **Total Records**: 6,698
- **Average Salary**: $115,329
- **Salary Std Dev**: $52,790
- **Average Experience**: 8.1 years

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone https://github.com/tugcesi/Salary_Prediction.git
cd Salary_Prediction
