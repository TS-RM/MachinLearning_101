# Data Transformation and Manipulation with Pandas

## Introduction to Data Transformation

Data transformation involves restructuring, combining, and manipulating data to make it more suitable for analysis. Pandas provides powerful tools for these operations, allowing you to reshape your data, create aggregations, apply functions, merge datasets, and work with time series data.

## Reshaping Data

Reshaping data involves changing the layout of a dataset without changing the data itself. Common reshaping operations include pivoting, melting, stacking, and unstacking.

### Pivot

The `pivot()` function reshapes data based on column values, transforming them into new columns:

```python
import pandas as pd
import numpy as np

# Create a sample DataFrame
df = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
    'product': ['A', 'B', 'A', 'B'],
    'sales': [100, 150, 120, 180]
})

# Pivot the data
pivot_df = df.pivot(index='date', columns='product', values='sales')
print(pivot_df)
```

Output:
```
product          A      B
date                     
2023-01-01  100.0  150.0
2023-01-02  120.0  180.0
```

If you have multiple values for the same combination of index and columns, you'll get an error. In such cases, you can use `pivot_table()`:

```python
# Create a DataFrame with duplicate entries
df = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-02'],
    'product': ['A', 'B', 'A', 'B'],
    'sales': [100, 150, 110, 180]
})

# Use pivot_table with an aggregation function
pivot_df = df.pivot_table(index='date', columns='product', values='sales', aggfunc='mean')
print(pivot_df)
```

### Melt

The `melt()` function is the inverse of pivot. It transforms columns into rows:

```python
# Create a wide-format DataFrame
wide_df = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-02'],
    'product_A': [100, 120],
    'product_B': [150, 180]
})

# Melt the DataFrame to long format
long_df = pd.melt(
    wide_df, 
    id_vars=['date'],
    value_vars=['product_A', 'product_B'],
    var_name='product',
    value_name='sales'
)
print(long_df)
```

Output:
```
         date    product  sales
0  2023-01-01  product_A    100
1  2023-01-02  product_A    120
2  2023-01-01  product_B    150
3  2023-01-02  product_B    180
```

### Stack and Unstack

`stack()` and `unstack()` are similar to melt and pivot but work with MultiIndex DataFrames:

```python
# Create a DataFrame with MultiIndex
multi_df = pd.DataFrame(
    data=np.random.randn(4, 2),
    index=pd.MultiIndex.from_product([['A', 'B'], [1, 2]], names=['first', 'second']),
    columns=['col1', 'col2']
)
print("Original DataFrame:")
print(multi_df)

# Stack the DataFrame (columns become rows)
stacked = multi_df.stack()
print("\nStacked DataFrame:")
print(stacked)

# Unstack the DataFrame (rows become columns)
unstacked = stacked.unstack()
print("\nUnstacked DataFrame (back to original):")
print(unstacked)

# Unstack at a different level
unstacked_level1 = stacked.unstack(level='first')
print("\nUnstacked at level 'first':")
print(unstacked_level1)
```

## Group and Aggregate Data

Grouping allows you to split data into groups based on some criteria, apply a function to each group independently, and combine the results.

### Basic Groupby Operations

```python
# Create a sample DataFrame
df = pd.DataFrame({
    'category': ['A', 'A', 'A', 'B', 'B', 'C'],
    'subcategory': ['X', 'Y', 'X', 'Y', 'Z', 'Z'],
    'value': [1, 2, 3, 4, 5, 6]
})

# Group by a single column and calculate mean
grouped = df.groupby('category')['value'].mean()
print(grouped)

# Group by multiple columns
grouped_multi = df.groupby(['category', 'subcategory'])['value'].mean()
print(grouped_multi)

# Convert the result to a DataFrame
grouped_df = grouped_multi.reset_index()
print(grouped_df)
```

### Multiple Aggregations

```python
# Apply multiple aggregation functions
agg_funcs = df.groupby('category')['value'].agg(['min', 'max', 'mean', 'count'])
print(agg_funcs)

# Different aggregations for different columns
agg_dict = df.groupby('category').agg({
    'value': ['min', 'max', 'mean'],
    'subcategory': 'nunique'  # Number of unique subcategories
})
print(agg_dict)
```

### Custom Aggregation Functions

```python
# Define a custom aggregation function
def range_func(x):
    return x.max() - x.min()

# Apply the custom function
custom_agg = df.groupby('category')['value'].agg(value_range=range_func)
print(custom_agg)

# Using lambda functions
custom_agg_lambda = df.groupby('category')['value'].agg(
    value_range=lambda x: x.max() - x.min(),
    first_last_diff=lambda x: x.iloc[-1] - x.iloc[0]
)
print(custom_agg_lambda)
```

### Transformation vs. Aggregation

While aggregation reduces the data, transformation returns a result with the same shape as the input:

```python
# Transformation: standardize values within each group
standardized = df.groupby('category')['value'].transform(lambda x: (x - x.mean()) / x.std())
df['value_standardized'] = standardized
print(df)

# Add group statistics to the original DataFrame
df['category_mean'] = df.groupby('category')['value'].transform('mean')
df['category_rank'] = df.groupby('category')['value'].transform('rank')
print(df)
```

## Apply Functions to Data

Pandas provides several methods to apply functions to data: `apply()`, `map()`, and `applymap()`.

### Using `apply()`

`apply()` applies a function along an axis of the DataFrame:

```python
# Create a sample DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

# Apply a function to each column
col_sums = df.apply(sum)
print(col_sums)

# Apply a function to each row
row_maxes = df.apply(max, axis=1)
print(row_maxes)

# Apply a more complex function
def range_func(x):
    return x.max() - x.min()

ranges = df.apply(range_func)
print(ranges)
```

### Using `map()`

`map()` is used for substituting each value in a Series:

```python
# Create a Series
s = pd.Series(['apple', 'banana', 'cherry'])

# Map values using a dictionary
fruit_colors = {'apple': 'red', 'banana': 'yellow', 'cherry': 'red'}
colors = s.map(fruit_colors)
print(colors)

# Map values using a function
lengths = s.map(len)
print(lengths)

# Handling missing values in mapping
fruit_colors = {'apple': 'red', 'banana': 'yellow'}  # Missing 'cherry'
colors = s.map(fruit_colors)  # 'cherry' will be mapped to NaN
print(colors)

# Provide a default value for missing mappings
colors = s.map(fruit_colors).fillna('unknown')
print(colors)
```

### Using `applymap()`

`applymap()` applies a function to every element in the DataFrame:

```python
# Create a DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

# Apply a function to every element
squared = df.applymap(lambda x: x**2)
print(squared)

# Format all values as strings with a prefix
formatted = df.applymap(lambda x: f"Value: {x}")
print(formatted)
```

## Merge and Join Datasets

Combining datasets is a common operation in data analysis. Pandas provides several methods for this: `merge()`, `join()`, and `concat()`.

### Using `merge()`

`merge()` combines DataFrames based on common columns or indices, similar to SQL joins:

```python
# Create two DataFrames
df1 = pd.DataFrame({
    'employee_id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'department_id': [101, 102, 101, 103]
})

df2 = pd.DataFrame({
    'department_id': [101, 102, 103, 104],
    'department_name': ['HR', 'Engineering', 'Finance', 'Marketing']
})

# Inner join (default)
inner_join = pd.merge(df1, df2, on='department_id')
print("Inner Join:")
print(inner_join)

# Left join
left_join = pd.merge(df1, df2, on='department_id', how='left')
print("\nLeft Join:")
print(left_join)

# Right join
right_join = pd.merge(df1, df2, on='department_id', how='right')
print("\nRight Join:")
print(right_join)

# Outer join
outer_join = pd.merge(df1, df2, on='department_id', how='outer')
print("\nOuter Join:")
print(outer_join)
```

### Merging on Different Column Names

```python
# Create DataFrames with different column names for the key
df1 = pd.DataFrame({
    'employee_id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'dept_id': [101, 102, 101, 103]  # Note the different name
})

df2 = pd.DataFrame({
    'department_id': [101, 102, 103, 104],  # Note the different name
    'department_name': ['HR', 'Engineering', 'Finance', 'Marketing']
})

# Merge on columns with different names
merged = pd.merge(df1, df2, left_on='dept_id', right_on='department_id')
print(merged)
```

### Using `join()`

`join()` is a convenient method for joining DataFrame objects by their indices:

```python
# Create DataFrames with meaningful indices
df1 = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'department_id': [101, 102, 101, 103]
}, index=[1, 2, 3, 4])

df2 = pd.DataFrame({
    'hire_date': ['2020-01-15', '2019-07-10', '2021-03-21', '2018-11-05'],
    'salary': [60000, 75000, 65000, 80000]
}, index=[1, 2, 3, 5])  # Note index 5 instead of 4

# Join on index
joined = df1.join(df2, how='inner')
print(joined)

# Left join
left_joined = df1.join(df2, how='left')
print(left_joined)
```

### Using `concat()`

`concat()` concatenates DataFrames along a particular axis:

```python
# Create DataFrames
df1 = pd.DataFrame({
    'A': [1, 2],
    'B': [3, 4]
})

df2 = pd.DataFrame({
    'A': [5, 6],
    'B': [7, 8]
})

# Concatenate vertically (along rows)
vertical_concat = pd.concat([df1, df2], axis=0)
print("Vertical Concatenation:")
print(vertical_concat)

# Reset index after concatenation
vertical_concat_reset = pd.concat([df1, df2], axis=0, ignore_index=True)
print("\nVertical Concatenation with Reset Index:")
print(vertical_concat_reset)

# Concatenate horizontally (along columns)
df3 = pd.DataFrame({
    'C': [9, 10],
    'D': [11, 12]
})

horizontal_concat = pd.concat([df1, df3], axis=1)
print("\nHorizontal Concatenation:")
print(horizontal_concat)
```

## Create Time Series Features

Time series data requires special handling. Pandas provides tools for working with dates, times, and time-based features.

### Creating Date Ranges

```python
# Create a date range
date_range = pd.date_range(start='2023-01-01', end='2023-01-10')
print(date_range)

# Create a date range with specific frequency
monthly = pd.date_range(start='2023-01-01', periods=12, freq='M')
print(monthly)

# Common frequencies
daily = pd.date_range(start='2023-01-01', periods=10, freq='D')
weekly = pd.date_range(start='2023-01-01', periods=10, freq='W')
business_days = pd.date_range(start='2023-01-01', periods=10, freq='B')
hourly = pd.date_range(start='2023-01-01', periods=24, freq='H')
```

### Extracting Date Components

```python
# Create a DataFrame with a date column
df = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=365, freq='D'),
    'value': np.random.randn(365)
})

# Extract date components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.day_name()
df['is_weekend'] = df['date'].dt.day_of_week >= 5  # 5=Saturday, 6=Sunday
df['quarter'] = df['date'].dt.quarter
df['is_month_end'] = df['date'].dt.is_month_end

print(df.head())
```

### Resampling Time Series Data

Resampling involves changing the frequency of your time series data:

```python
# Set the date as index
df.set_index('date', inplace=True)

# Downsample to monthly frequency
monthly_mean = df['value'].resample('M').mean()
print(monthly_mean)

# Downsample to weekly frequency with multiple aggregations
weekly_stats = df['value'].resample('W').agg(['mean', 'min', 'max', 'std'])
print(weekly_stats)

# Upsample to hourly frequency and forward fill
hourly = df['value'].resample('H').ffill()
```

### Shifting and Lagging

Shifting data is useful for calculating changes and creating lag features:

```python
# Create a lag feature (previous day's value)
df['value_lag1'] = df['value'].shift(1)

# Create a lead feature (next day's value)
df['value_lead1'] = df['value'].shift(-1)

# Calculate day-over-day change
df['value_change'] = df['value'] - df['value_lag1']

# Calculate percentage change
df['value_pct_change'] = df['value'].pct_change() * 100

print(df.head())
```

### Rolling Windows

Rolling windows are useful for calculating moving averages and other statistics:

```python
# Calculate 7-day moving average
df['value_7d_ma'] = df['value'].rolling(window=7).mean()

# Calculate 30-day moving standard deviation
df['value_30d_std'] = df['value'].rolling(window=30).std()

# Calculate expanding (cumulative) statistics
df['value_expanding_mean'] = df['value'].expanding().mean()

# Calculate exponentially weighted moving average
df['value_ewm'] = df['value'].ewm(span=7).mean()

print(df.head(10))
```

## Comprehensive Data Transformation Example

Let's put everything together with a comprehensive example:

```python
import pandas as pd
import numpy as np
from datetime import datetime

# Create a sample sales dataset
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-12-31')
products = ['Product A', 'Product B', 'Product C']
regions = ['North', 'South', 'East', 'West']

# Generate random data
data = []
for date in dates:
    for product in products:
        for region in regions:
            sales = np.random.randint(10, 100)
            price = np.random.choice([9.99, 19.99, 29.99, 39.99])
            data.append([date, product, region, sales, price])

# Create DataFrame
df = pd.DataFrame(data, columns=['date', 'product', 'region', 'units_sold', 'unit_price'])

# Calculate revenue
df['revenue'] = df['units_sold'] * df['unit_price']

# 1. Group and aggregate data
# Monthly sales by product
monthly_product_sales = df.groupby([pd.Grouper(key='date', freq='M'), 'product'])['revenue'].sum().reset_index()
monthly_product_sales['month'] = monthly_product_sales['date'].dt.strftime('%Y-%m')

# Regional performance
regional_performance = df.groupby('region').agg({
    'units_sold': 'sum',
    'revenue': 'sum'
}).reset_index()
regional_performance['average_price'] = regional_performance['revenue'] / regional_performance['units_sold']

# 2. Reshape data for analysis
# Pivot table of monthly product revenue
pivot_monthly_revenue = monthly_product_sales.pivot(index='month', columns='product', values='revenue')

# 3. Time series features
# Set date as index
df_ts = df.copy()
df_ts['date'] = pd.to_datetime(df_ts['date'])
df_ts.set_index('date', inplace=True)

# Daily total sales
daily_sales = df_ts.groupby(pd.Grouper(freq='D'))['revenue'].sum()

# Add time features
daily_sales = daily_sales.reset_index()
daily_sales['day_of_week'] = daily_sales['date'].dt.day_name()
daily_sales['is_weekend'] = daily_sales['date'].dt.day_of_week >= 5
daily_sales['month'] = daily_sales['date'].dt.month
daily_sales['quarter'] = daily_sales['date'].dt.quarter

# Calculate 7-day moving average
daily_sales['revenue_7d_ma'] = daily_sales['revenue'].rolling(window=7).mean()

# 4. Merge datasets
# Create a product information dataset
product_info = pd.DataFrame({
    'product': products,
    'category': ['Electronics', 'Clothing', 'Home'],
    'supplier': ['Supplier X', 'Supplier Y', 'Supplier Z']
})

# Merge with monthly product sales
product_sales_info = pd.merge(monthly_product_sales, product_info, on='product')

# 5. Apply functions
# Calculate revenue contribution percentage
total_revenue = df['revenue'].sum()
df['revenue_pct'] = df['revenue'].apply(lambda x: (x / total_revenue) * 100)

# Categorize sales
def categorize_sales(units):
    if units < 30:
        return 'Low'
    elif units < 70:
        return 'Medium'
    else:
        return 'High'

df['sales_category'] = df['units_sold'].apply(categorize_sales)

# Print results
print("Monthly Product Sales:")
print(monthly_product_sales.head())

print("\nRegional Performance:")
print(regional_performance)

print("\nPivot Table of Monthly Product Revenue:")
print(pivot_monthly_revenue.head())

print("\nDaily Sales with Moving Average:")
print(daily_sales.head())

print("\nProduct Sales with Category Information:")
print(product_sales_info.head())

print("\nSales Categorization:")
print(df[['product', 'units_sold', 'sales_category']].head(10))
```

## Practice Exercises

1. Create a dataset with sales data and reshape it using pivot and melt functions.
2. Group a dataset by multiple columns and calculate various aggregations.
3. Create a function to categorize values in a dataset and apply it using different methods.
4. Merge two datasets with different join types and compare the results.
5. Create a time series dataset and extract useful features from the date information.

## Key Takeaways

- Reshaping functions like pivot, melt, stack, and unstack help transform data between wide and long formats
- Groupby operations allow you to split, apply, and combine data based on specific criteria
- Apply, map, and applymap provide flexible ways to transform data using custom functions
- Merge, join, and concat enable you to combine multiple datasets in various ways
- Time series features and resampling help extract valuable insights from date-based data
- Mastering these transformation techniques is essential for effective data analysis and preparation for visualization
