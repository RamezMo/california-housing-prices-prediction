# Stroke-Predictor-Model : Machine Learning Model for Forecasting Stroce with nearly 98% accuracy

## Introduction

Accurately predicting housing prices is crucial for buyers, sellers, and investors in the real estate market. This project applies machine learning techniques to forecast housing prices in California using the California Housing Prices dataset from Kaggle. By examining factors such as median income, average number of rooms, average number of bedrooms, population, and geographical data, the model aims to deliver precise price predictions. This tool is designed to support stakeholders in making informed real estate decisions, enhancing investment strategies, and understanding market trends.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Preparation](#Data-Preparation)
3. [Exploratory Data Analysis (EDA)](#Exploratory-Data-Analysis) 
4. [Model Evaluation](#Evaluate-models)

# Data Preparation

### Importing Necessary Libraries

Begin by importing the essential libraries for data analysis and machine learning.

```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.datasets
from xgboost import XGBRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
```

## Load dataset using full path
Load the dataset into DataFrame.

```python
df = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')
```

## Display the first few rows of the dataset
Review the initial rows to comprehend the dataset's structure.
```python
df.head()
```

![image](https://github.com/RamezMo/california-housing-prices-prediction/assets/142325393/4ae77ebd-34c4-4101-a79d-2c4449ae77c1)



## How Many Instances and Features ?
Check How many Rows and Columns do we have
```python
df.shape
```
#Exploratory-Data-Analysis

## Checking for missing values
Check for and handle missing values to ensure a clean dataset.

```python
df.isna().sum()
```

![image](https://github.com/RamezMo/california-housing-prices-prediction/assets/142325393/4a46d761-6c52-4a8e-8a58-dbcf822449bc)


## Replace rows with missing values (with mean Value since it is a numerical column)

```python
df.total_bedrooms.fillna(df.total_bedrooms.mean(),inplace=True)
```

##Display Variables DataType and number of non-NULLs values in each Variable

```python
df.info()
```

![image](https://github.com/RamezMo/california-housing-prices-prediction/assets/142325393/cd7f7e7f-ee6f-432a-8afb-552d7c385fd2)

## Summary statistics
Discover the Min , Max and Mean Values for each Column and How Data Distributed over Quantiles
```python
print(df.describe())
```
## Key Observations
* Age: The average age For Houses is 	28.6 years
* Average Prices: On average Houses price is 	206855 and it begins with 	14999 and the Maximum House Price is 	500001


#data preperation
## Data Transformation

Convert categorical variables into numerical ones for machine learning models.

```python
df_dummies = pd.get_dummies(df['ocean_proximity'], prefix='ocean_proximity').astype(int)

# Add the new dummy columns to the original DataFrame
df = pd.concat([df, df_dummies], axis=1)

# Display the modified DataFrame
df.head()
```



![image](https://github.com/RamezMo/california-housing-prices-prediction/assets/142325393/bdccc26e-def5-43f6-a25c-57bc8735b674)

## Calculate the correlation matrix

```python
corr=df.corr()
```
## Create a correlation heatmap for the subset of features
this Shows a strong positive correlation Between median-house-price and median-income
```python
plt.figure(figsize=(20,20))
sns.heatmap(corr,annot=True,mask = np.triu(np.ones_like(corr, dtype=bool)))
```

## Split data into training and testing sets

```python
x=df.drop('median_house_value',axis=1)
y=df.median_house_value
```


## Import LazyPredict 
it is a library that is used to evaluate many algorithms and find the accuracies for each one
in this case we find that XGBRegressor achieves the best accuracy 
```python
from lazypredict.Supervised import LazyRegressor
regressor = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = regressor.fit(x_train, x_test, y_train, y_test)

```


## Evaluate models
After training the model and predicting on the test data, it achieved an R² score of 0.833, indicating a strong fit. 
The Mean Absolute Error (MAE) for testing was 31,186.46, reflecting the average prediction error in house prices.
![image](https://github.com/RamezMo/california-housing-prices-prediction/assets/142325393/d66ce05e-e329-4b83-b4d4-3d5586193a74)


##Creating "Actual vs. Predicted Values" Plot
#Plot actual vs. predicted values For Testing Data
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predicted_test, alpha=0.7, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel('Actual-Test')
plt.ylabel('Predicted-Test')
plt.title('Actual vs. Predicted Values')
plt.show()
```
![image](https://github.com/RamezMo/california-housing-prices-prediction/assets/142325393/2e39e26d-6665-4fbb-bd62-8e0eac670846)
