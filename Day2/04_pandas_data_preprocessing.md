# Data Preprocessing Methods with Pandas

## Introduction to Data Preprocessing

Data preprocessing is the crucial step between data cleaning and analysis or modeling. While data cleaning focuses on fixing errors and inconsistencies, preprocessing transforms the data into a format that's more suitable for analysis and machine learning algorithms.

## Normalizing and Standardizing Data

Normalization and standardization are techniques to scale numeric features to a similar range, which is important for many machine learning algorithms.

### Min-Max Normalization

Min-max normalization scales values to a range between 0 and 1:

```python
import pandas as pd
import numpy as np

# Create a sample DataFrame
df = pd.DataFrame({
    'A': [1, 5, 10, 15, 20],
    'B': [100, 200, 300, 400, 500]
})

# Min-max normalization
def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# Apply to each numeric column
df_normalized = df.apply(min_max_normalize)

print(df_normalized)
```

### Z-Score Standardization

Z-score standardization scales data to have a mean of 0 and a standard deviation of 1:

```python
# Z-score standardization
def z_score_standardize(series):
    return (series - series.mean()) / series.std()

# Apply to each numeric column
df_standardized = df.apply(z_score_standardize)

print(df_standardized)
```

### Using Scikit-learn for Scaling

Scikit-learn provides scalers that can be more convenient, especially when you need to apply the same transformation to training and test data:

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Min-max scaling
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Standardization
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
```

## One-Hot Encoding

One-hot encoding converts categorical variables into a format that works better with machine learning algorithms.

### Using Pandas get_dummies

```python
# Create a DataFrame with categorical variables
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['small', 'medium', 'large', 'medium', 'small']
})

# One-hot encode all categorical columns
df_encoded = pd.get_dummies(df)
print(df_encoded)

# One-hot encode specific columns
df_encoded = pd.get_dummies(df, columns=['color'])
print(df_encoded)

# Add a prefix to avoid name collisions
df_encoded = pd.get_dummies(df, prefix=['col', 'sz'])
print(df_encoded)
```

### Handling Unknown Categories

When applying one-hot encoding to test data, you might encounter categories not seen in training:

```python
# Create training and test data
train_df = pd.DataFrame({'color': ['red', 'blue', 'green']})
test_df = pd.DataFrame({'color': ['red', 'yellow', 'blue']})

# Get all unique categories
all_categories = pd.concat([train_df['color'], test_df['color']]).unique()

# One-hot encode with all possible categories
train_encoded = pd.get_dummies(train_df['color'], prefix='color')
test_encoded = pd.get_dummies(test_df['color'], prefix='color')

# Ensure test data has same columns as training data
for category in all_categories:
    col_name = f'color_{category}'
    if col_name not in train_encoded.columns:
        train_encoded[col_name] = 0
    if col_name not in test_encoded.columns:
        test_encoded[col_name] = 0

# Reorder columns to match
test_encoded = test_encoded[train_encoded.columns]
```

### Using Scikit-learn for One-Hot Encoding

```python
from sklearn.preprocessing import OneHotEncoder

# Initialize the encoder
encoder = OneHotEncoder(sparse=False)

# Fit and transform
encoded_data = encoder.fit_transform(df[['color']])

# Convert to DataFrame
encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(['color'])
)

# Concatenate with original data
result = pd.concat([df.drop('color', axis=1), encoded_df], axis=1)
```

## Creating Binary Features

Binary features are special cases of categorical features with only two possible values.

### From Categorical Variables

```python
# Create a binary feature from a categorical variable
df['is_red'] = (df['color'] == 'red').astype(int)

# Create multiple binary features
colors = ['red', 'blue', 'green']
for color in colors:
    df[f'is_{color}'] = (df['color'] == color).astype(int)
```

### From Numeric Thresholds

```python
# Create a DataFrame with numeric data
df = pd.DataFrame({
    'age': [25, 35, 45, 55, 65],
    'income': [30000, 45000, 60000, 75000, 90000]
})

# Create binary features based on thresholds
df['is_young'] = (df['age'] < 30).astype(int)
df['is_high_income'] = (df['income'] > 50000).astype(int)
```

### From Text Data

```python
# Create a DataFrame with text data
df = pd.DataFrame({
    'product_review': [
        'Great product, highly recommended!',
        'Terrible experience, would not buy again.',
        'Good value for money, satisfied.',
        'Product broke after one week, disappointed.'
    ]
})

# Create binary features based on text content
df['is_positive'] = df['product_review'].str.contains('great|good|recommended|satisfied', case=False).astype(int)
df['is_negative'] = df['product_review'].str.contains('terrible|broke|disappointed|not', case=False).astype(int)
```

## Binning Continuous Variables

Binning (or discretization) converts continuous variables into categorical ones by dividing the range into intervals.

### Equal-Width Binning

```python
# Create a DataFrame with continuous data
df = pd.DataFrame({
    'age': [22, 35, 46, 58, 65, 72, 81, 29, 37, 51]
})

# Create bins with equal width
df['age_group'] = pd.cut(df['age'], bins=3)
print(df)

# Specify custom bin edges
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 60, 100])
print(df)

# Specify custom bin labels
df['age_group'] = pd.cut(
    df['age'], 
    bins=[0, 30, 60, 100], 
    labels=['Young', 'Middle-aged', 'Senior']
)
print(df)
```

### Equal-Frequency Binning (Quantiles)

```python
# Create bins with equal number of samples
df['age_quantile'] = pd.qcut(df['age'], q=4)  # Quartiles
print(df)

# Specify custom quantiles
df['age_quantile'] = pd.qcut(df['age'], q=[0, 0.25, 0.75, 1.0])
print(df)

# Specify custom labels
df['age_quantile'] = pd.qcut(
    df['age'], 
    q=[0, 0.25, 0.75, 1.0], 
    labels=['Bottom 25%', 'Middle 50%', 'Top 25%']
)
print(df)
```

### One-Hot Encoding Binned Variables

After binning, you often want to one-hot encode the resulting categories:

```python
# Bin the data
df['age_group'] = pd.cut(
    df['age'], 
    bins=[0, 30, 60, 100], 
    labels=['Young', 'Middle-aged', 'Senior']
)

# One-hot encode the binned variable
age_dummies = pd.get_dummies(df['age_group'], prefix='age')

# Concatenate with original data
df = pd.concat([df, age_dummies], axis=1)
```

## Scaling Features for Machine Learning

Different machine learning algorithms have different requirements for feature scaling.

### When to Scale Features

- **Algorithms that use distances**: K-means clustering, K-nearest neighbors, Support Vector Machines
- **Algorithms with regularization**: Ridge and Lasso regression, neural networks
- **Gradient-based optimization**: Most neural networks and deep learning models

### Scaling Techniques

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

# Create a DataFrame with numeric features
df = pd.DataFrame({
    'feature1': [1, 5, 10, 15, 20],
    'feature2': [100, 200, 300, 400, 500],
    'feature3': [2, 2, 3, 3, 5]
})

# 1. StandardScaler (Z-score normalization)
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)

# 2. MinMaxScaler (scales to a specific range, default 0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)

# 3. RobustScaler (uses median and quantiles, robust to outliers)
scaler = RobustScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)

# 4. MaxAbsScaler (scales by maximum absolute value)
scaler = MaxAbsScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)
```

### Scaling in a Machine Learning Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Prepare data
X = df.drop('target', axis=1)  # Features
y = df['target']  # Target variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # First scale the features
    ('classifier', LogisticRegression())  # Then apply the classifier
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)
```

## Comprehensive Preprocessing Example

Let's put everything together with a comprehensive example:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv('data.csv')

# Identify numeric and categorical columns
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Remove target variable from features
target = 'target_variable'
if target in numeric_features:
    numeric_features.remove(target)
if target in categorical_features:
    categorical_features.remove(target)

# Create preprocessing pipelines for both numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the preprocessing and modeling pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Split data
X = df.drop(target, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline
model_pipeline.fit(X_train, y_train)

# Make predictions
predictions = model_pipeline.predict(X_test)
```

## Practice Exercises

1. Find a dataset with numeric features and apply different scaling techniques. Compare the results visually.
2. Create a dataset with categorical variables and practice one-hot encoding.
3. Take a continuous variable and bin it using both equal-width and equal-frequency binning. Compare the distributions.
4. Create binary features from different data types (categorical, numeric, text).
5. Build a complete preprocessing pipeline for a dataset with mixed data types.

## Key Takeaways

- Data preprocessing transforms cleaned data into a format suitable for analysis and modeling
- Scaling techniques like normalization and standardization are essential for many algorithms
- One-hot encoding converts categorical variables into a machine-learning-friendly format
- Binary features can be created from various data types and are useful for many analyses
- Binning continuous variables can reveal patterns and simplify complex relationships
- Scikit-learn's Pipeline and ColumnTransformer make it easy to build reproducible preprocessing workflows
