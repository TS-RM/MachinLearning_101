# Data Cleaning Techniques with Pandas

## Introduction to Data Cleaning

Data cleaning is a critical step in the data analysis process. Real-world data is often messy, incomplete, and inconsistent. Effective data cleaning ensures that your analysis is based on accurate and reliable information.

## Handling Missing Values

Missing values are a common issue in datasets. Pandas provides several methods to detect and handle them.

### Detecting Missing Values

```python
import pandas as pd
import numpy as np

# Create a DataFrame with missing values
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': ['a', 'b', 'c', None]
})

# Check for missing values
print(df.isna())  # Returns a boolean DataFrame where True indicates missing values

# Count missing values in each column
print(df.isna().sum())

# Check if any value in a row is missing
print(df.isna().any(axis=1))

# Get percentage of missing values per column
print(df.isna().mean() * 100)
```

### Dropping Missing Values

The `dropna()` method removes rows or columns with missing values:

```python
# Drop rows with any missing values
df_cleaned = df.dropna()

# Drop rows only if all values are missing
df_cleaned = df.dropna(how='all')

# Drop rows that have fewer than 2 non-NA values
df_cleaned = df.dropna(thresh=2)

# Drop columns with any missing values
df_cleaned = df.dropna(axis=1)
```

### Filling Missing Values

The `fillna()` method replaces missing values:

```python
# Fill all missing values with a specific value
df_filled = df.fillna(0)  # Fill with zero
df_filled = df.fillna('Unknown')  # Fill with string

# Fill with different values for each column
df_filled = df.fillna({'A': 0, 'B': 5, 'C': 'Unknown'})

# Fill with the mean of each column
df_filled = df.fillna(df.mean())  # Only works for numeric columns

# Forward fill (use the previous value)
df_filled = df.fillna(method='ffill')

# Backward fill (use the next value)
df_filled = df.fillna(method='bfill')

# Interpolate values (linear interpolation)
df_filled = df.interpolate()
```

### Choosing the Right Strategy

The best approach depends on your data and analysis goals:

- **Drop rows**: When missing data represents a small portion of your dataset
- **Fill with statistics**: When you want to preserve the overall distribution (mean, median, mode)
- **Forward/backward fill**: For time series data where values are related to previous/next values
- **Interpolation**: For numerical data where you want to estimate missing values based on surrounding data
- **Fill with a specific value**: When missing values have a meaningful interpretation (e.g., 0 for no sales)

## Removing Duplicates

Duplicate data can skew your analysis results. Pandas provides tools to identify and remove duplicates.

### Identifying Duplicates

```python
# Create a DataFrame with duplicates
df = pd.DataFrame({
    'A': [1, 2, 2, 3, 3],
    'B': ['a', 'b', 'b', 'c', 'c']
})

# Check for duplicate rows
print(df.duplicated())  # Returns a boolean Series where True indicates duplicates

# Count duplicates
print(df.duplicated().sum())
```

### Removing Duplicates

```python
# Remove duplicate rows (keeps first occurrence)
df_unique = df.drop_duplicates()

# Remove duplicates, keeping last occurrence
df_unique = df.drop_duplicates(keep='last')

# Remove duplicates, dropping all occurrences
df_unique = df.drop_duplicates(keep=False)

# Remove duplicates based on specific columns
df_unique = df.drop_duplicates(subset=['A'])
```

## Fixing Data Types

Incorrect data types can cause errors and unexpected behavior. Pandas provides methods to check and convert data types.

### Checking Data Types

```python
# Check data types of all columns
print(df.dtypes)

# Get detailed information
print(df.info())
```

### Converting Data Types

```python
# Convert a column to a different type
df['A'] = df['A'].astype('float')
df['B'] = df['B'].astype('category')  # For categorical data
df['C'] = df['C'].astype('str')

# Convert multiple columns
df = df.astype({'A': 'float', 'B': 'category', 'C': 'str'})

# Convert to numeric, coercing errors to NaN
df['D'] = pd.to_numeric(df['D'], errors='coerce')

# Convert to datetime
df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
```

### Handling Mixed Types

Sometimes columns contain mixed data types:

```python
# Example: Column with mixed numeric and string values
mixed_data = ['1', '2', 'three', '4', 'five']
series = pd.Series(mixed_data)

# Convert to numeric, coercing errors to NaN
numeric_series = pd.to_numeric(series, errors='coerce')

# Then handle the NaN values
numeric_series = numeric_series.fillna(0)
```

## String Manipulation and Cleaning

Text data often requires cleaning and standardization. Pandas provides string methods through the `.str` accessor.

### Basic String Operations

```python
# Create a Series with string data
s = pd.Series(['  John Smith  ', 'JANE DOE', 'robert johnson', np.nan])

# Remove leading/trailing whitespace
s = s.str.strip()

# Convert to lowercase/uppercase
s = s.str.lower()
s = s.str.upper()
s = s.str.title()  # Title Case

# Check if string contains a pattern
mask = s.str.contains('john', case=False)

# Replace text
s = s.str.replace('John', 'Jonathan')
s = s.str.replace(r'\s+', ' ', regex=True)  # Replace multiple spaces with single space
```

### Extracting Information

```python
# Extract part of a string
s = pd.Series(['John_Smith', 'Jane_Doe', 'Robert_Johnson'])
first_names = s.str.split('_').str[0]
last_names = s.str.split('_').str[1]

# Extract using regular expressions
emails = pd.Series(['john@example.com', 'jane@company.org'])
domains = emails.str.extract(r'@(.+)$')

# Extract multiple groups
names = pd.Series(['John Smith', 'Jane Doe'])
name_parts = names.str.extract(r'(\w+)\s+(\w+)')  # Returns DataFrame with first and last names
```

### Handling Missing Values in String Operations

String methods automatically skip NaN values:

```python
# Series with missing values
s = pd.Series(['apple', 'banana', np.nan, 'cherry'])

# String operations skip NaN
lowercase = s.str.lower()  # NaN remains NaN
```

## Handling Outliers and Anomalies

Outliers can significantly impact statistical analyses and machine learning models.

### Detecting Outliers

#### Using Statistical Methods

```python
# Z-score method
from scipy import stats
import numpy as np

z_scores = stats.zscore(df['value'])
abs_z_scores = np.abs(z_scores)
outliers = df[abs_z_scores > 3]  # Values more than 3 standard deviations away

# IQR method
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
```

#### Visualization for Outlier Detection

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['value'])
plt.title('Box Plot for Outlier Detection')
plt.show()

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df.index, df['value'])
plt.title('Scatter Plot for Outlier Detection')
plt.show()
```

### Handling Outliers

The approach depends on your analysis goals:

```python
# 1. Remove outliers
df_cleaned = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]

# 2. Cap outliers (Winsorization)
df['value_capped'] = df['value'].clip(lower=lower_bound, upper=upper_bound)

# 3. Replace with NaN and then handle as missing values
df.loc[(df['value'] < lower_bound) | (df['value'] > upper_bound), 'value'] = np.nan
df['value'] = df['value'].fillna(df['value'].median())

# 4. Log transformation to reduce the impact of outliers
df['value_log'] = np.log1p(df['value'])  # log1p = log(1+x) to handle zeros
```

## Comprehensive Data Cleaning Example

Let's put everything together with a comprehensive example:

```python
import pandas as pd
import numpy as np
from scipy import stats

# Load a messy dataset
df = pd.read_csv('messy_data.csv')

# 1. Examine the data
print(df.head())
print(df.info())
print(df.describe())

# 2. Handle missing values
# Check missing values
missing_values = df.isna().sum()
print(f"Missing values per column:\n{missing_values}")

# Fill numeric columns with median
numeric_cols = df.select_dtypes(include=['number']).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 3. Fix data types
# Convert date columns to datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Convert numeric columns that might be stored as strings
for col in ['age', 'income']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 4. Clean text data
# Standardize text columns
if 'name' in df.columns:
    df['name'] = df['name'].str.strip().str.title()

# 5. Remove duplicates
df = df.drop_duplicates()

# 6. Handle outliers
# Identify numeric columns for outlier detection
numeric_cols = df.select_dtypes(include=['number']).columns

for col in numeric_cols:
    # Calculate IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

# 7. Final check
print("\nCleaned data info:")
print(df.info())
print(df.describe())

# 8. Save cleaned data
df.to_csv('cleaned_data.csv', index=False)
```

## Practice Exercises

1. Find a dataset with missing values and apply different strategies to handle them. Compare the results.
2. Create a dataset with outliers and practice detecting and handling them using different methods.
3. Clean a dataset with messy string values (e.g., inconsistent capitalization, extra spaces).
4. Find a dataset with mixed data types and fix them.
5. Create a complete data cleaning pipeline for a real-world dataset.

## Key Takeaways

- Data cleaning is a crucial step in the data analysis process
- Pandas provides powerful tools for handling missing values, duplicates, and data type issues
- String methods in Pandas make text data cleaning straightforward
- Outlier detection and handling requires both statistical knowledge and domain expertise
- A systematic approach to data cleaning ensures reliable analysis results
