#  Titanic Survival Prediction using Logistic Regression

This project aims to predict passenger survival on the Titanic using logistic regression. It is based on the famous Titanic dataset provided by Kaggle. The model has been trained using essential data preprocessing, label encoding, and machine learning techniques with `scikit-learn`.

---

## Overview

- Dataset: Titanic dataset (`titanic.csv`)
- Algorithm: Logistic Regression
- Accuracy: ~77%
- Tools: Python, Pandas, scikit-learn, Matplotlib, Pickle/Joblib

---

## Dataset Features

| Feature       | Description                                      |
|---------------|--------------------------------------------------|
| PassengerId   | Unique ID for each passenger                     |
| Survived      | Target variable (0 = No, 1 = Yes)                |
| Pclass        | Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)       |
| Name          | Passenger name                                   |
| Sex           | Gender                                           |
| Age           | Age in years                                     |
| SibSp         | # of siblings/spouses aboard                     |
| Parch         | # of parents/children aboard                     |
| Ticket        | Ticket number                                    |
| Fare          | Ticket fare                                      |
| Cabin         | Cabin number                                     |
| Embarked      | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## Data Preprocessing

- Handled missing values:
  - `Age`: Replaced with median
  - `Cabin` and `Embarked`: Replaced with mode
- Converted categorical columns (`Sex`, `Embarked`, `Cabin`, etc.) to numerical using `LabelEncoder`.

---

## Model Training

- **Train-test split**: 80% training, 20% testing
- **Classifier used**: `LogisticRegression()` from `sklearn.linear_model`
- **Accuracy**: `77.09%` on test data

---

## Model Saving

You can save the trained model using either `pickle` or `joblib`:

```python
# Using pickle
import pickle
with open('LogisticReg.pkl', 'wb') as f:
    pickle.dump(model, f)

# Using joblib
import joblib
joblib.dump(model, 'LogisticReg.pkl')
