# Pandas Basics

## Introduction to Pandas

Pandas is a powerful Python library for data manipulation and analysis. It provides data structures and functions needed to efficiently work with structured data. The name "Pandas" is derived from "panel data," an econometrics term for multidimensional structured datasets.

### Installing Pandas

Before you can use Pandas, you need to install it. You can install Pandas using pip:

```python
pip install pandas
```

To verify your installation, run:

```python
import pandas as pd
print(pd.__version__)
```

### The Pandas Ecosystem

Pandas works well with other libraries in the Python data science ecosystem:

- **NumPy**: Provides the fundamental array data structure that Pandas is built upon
- **Matplotlib**: For data visualization
- **Scikit-learn**: For machine learning algorithms
- **SciPy**: For scientific computing
- **Statsmodels**: For statistical models

## Core Data Structures

Pandas has two primary data structures:

### 1. Series

A Series is a one-dimensional labeled array capable of holding any data type. It's similar to a column in a spreadsheet or a single variable in statistics.

```python
import pandas as pd
import numpy as np

# Creating a Series from a list
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)
```

Output:
```
0    1.0
1    3.0
2    5.0
3    NaN
4    6.0
5    8.0
dtype: float64
```

Series have an index that labels each element:

```python
# Creating a Series with a custom index
s = pd.Series([1, 3, 5, 7, 9], index=['a', 'b', 'c', 'd', 'e'])
print(s)
```

Output:
```
a    1
b    3
c    5
d    7
e    9
dtype: int64
```

### 2. DataFrame

A DataFrame is a two-dimensional labeled data structure with columns that can be of different types. It's similar to a spreadsheet, SQL table, or a dictionary of Series objects.

```python
# Creating a DataFrame from a dictionary
data = {
    'Name': ['John', 'Anna', 'Peter', 'Linda'],
    'Age': [28, 34, 29, 42],
    'City': ['New York', 'Paris', 'Berlin', 'London']
}

df = pd.DataFrame(data)
print(df)
```

Output:
```
    Name  Age      City
0   John   28  New York
1   Anna   34     Paris
2  Peter   29    Berlin
3  Linda   42    London
```

You can also create a DataFrame from various other sources:

```python
# From a list of dictionaries
data = [
    {'Name': 'John', 'Age': 28},
    {'Name': 'Anna', 'Age': 34, 'City': 'Paris'},
    {'Name': 'Peter', 'Age': 29}
]
df = pd.DataFrame(data)

# From a NumPy array
df = pd.DataFrame(np.random.randn(5, 3), columns=['A', 'B', 'C'])
```

## Indexing and Selection Methods

Pandas provides multiple ways to select and access data:

### Using `[]` Operator

```python
# Select a single column (returns a Series)
names = df['Name']

# Select multiple columns (returns a DataFrame)
subset = df[['Name', 'Age']]

# Select rows by position
first_two_rows = df[0:2]
```

### Using `loc` (Label-based)

`loc` is used for label-based indexing:

```python
# Select a single value by row and column label
value = df.loc[0, 'Name']  # 'John'

# Select multiple rows and columns by label
subset = df.loc[0:2, ['Name', 'Age']]

# Select all rows for specific columns
names_only = df.loc[:, 'Name']

# Boolean indexing
adults = df.loc[df['Age'] > 30]
```

### Using `iloc` (Integer-based)

`iloc` is used for integer-based indexing:

```python
# Select by integer position
first_cell = df.iloc[0, 0]  # First row, first column

# Select multiple rows and columns by position
subset = df.iloc[0:2, 0:2]

# Select all rows for specific columns
first_col = df.iloc[:, 0]
```

### Using `at` and `iat` (Fast Scalar Access)

For fast access to a single value:

```python
# Label-based scalar lookup
value = df.at[0, 'Name']

# Integer-based scalar lookup
value = df.iat[0, 0]
```

## Data Types and Basic Operations

### Checking Data Types

```python
# Check the data types of all columns
print(df.dtypes)

# Get detailed information about the DataFrame
print(df.info())
```

### Basic Operations

```python
# Arithmetic operations
df['Age_in_months'] = df['Age'] * 12

# Statistical operations
mean_age = df['Age'].mean()
median_age = df['Age'].median()
age_stats = df['Age'].describe()

# Applying functions
df['Name_length'] = df['Name'].apply(len)

# Sorting
df_sorted = df.sort_values(by='Age', ascending=False)
```

## Built-in Functions

Pandas provides many useful functions for exploring and summarizing data:

### `head()` and `tail()`

```python
# View the first 5 rows
print(df.head())

# View the last 3 rows
print(df.tail(3))
```

### `info()`

```python
# Get a concise summary of the DataFrame
print(df.info())
```

Output:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4 entries, 0 to 3
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   Name    4 non-null      object
 1   Age     4 non-null      int64 
 2   City    4 non-null      object
dtypes: int64(1), object(2)
memory usage: 224.0+ bytes
```

### `describe()`

```python
# Generate descriptive statistics
print(df.describe())
```

Output:
```
             Age
count   4.000000
mean   33.250000
std     6.397656
min    28.000000
25%    28.750000
50%    31.500000
75%    36.000000
max    42.000000
```

By default, `describe()` only includes numeric columns. To include all columns:

```python
print(df.describe(include='all'))
```

### `value_counts()`

```python
# Count unique values in a column
print(df['City'].value_counts())
```

## Practice Exercises

1. Create a Series with your favorite foods as values and numbers 1-5 as the index.
2. Create a DataFrame with information about 5 books (title, author, year, rating).
3. Use different selection methods to:
   - Get the title of the third book
   - Get all books published after 2000
   - Get titles and authors of the first two books
4. Calculate the average rating of all books.
5. Add a new column indicating whether the book is "Recent" (published after 2010) or "Older".

## Key Takeaways

- Pandas provides two main data structures: Series (1D) and DataFrame (2D)
- There are multiple ways to select data: [], loc, iloc, at, iat
- Basic operations include arithmetic, statistical functions, and sorting
- Built-in functions like head(), info(), and describe() help explore data
- Understanding these basics is essential before moving to more complex data manipulation tasks
