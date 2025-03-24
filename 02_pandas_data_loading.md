# Data Loading with Pandas

## Introduction to Data Loading

One of Pandas' most powerful features is its ability to read data from various file formats and data sources. This capability allows you to work with data regardless of how it's stored, making Pandas an essential tool for data analysis.

## Loading Data from CSV Files

CSV (Comma-Separated Values) is one of the most common formats for storing tabular data. Pandas makes it easy to read CSV files using the `read_csv()` function.

### Basic CSV Loading

```python
import pandas as pd

# Basic usage
df = pd.read_csv('data.csv')
print(df.head())
```

### Customizing CSV Import

The `read_csv()` function has many parameters to handle different CSV formats:

```python
# Specify delimiter (for tab-separated files)
df = pd.read_csv('data.tsv', delimiter='\t')

# Skip rows
df = pd.read_csv('data.csv', skiprows=2)  # Skip first 2 rows

# Specify column names
df = pd.read_csv('data.csv', names=['ID', 'Name', 'Value'])

# Use specific columns as index
df = pd.read_csv('data.csv', index_col='ID')

# Handle missing values
df = pd.read_csv('data.csv', na_values=['NA', 'Missing', '?'])
```

### Reading Specific Chunks

For large files, you can read data in chunks:

```python
# Read only first 1000 rows
df = pd.read_csv('large_file.csv', nrows=1000)

# Read file in chunks
chunks = pd.read_csv('large_file.csv', chunksize=10000)
for chunk in chunks:
    # Process each chunk
    process_data(chunk)
```

## Loading Data from Excel Files

Excel files are widely used in business settings. Pandas can read Excel files using the `read_excel()` function.

### Basic Excel Loading

```python
# Basic usage
df = pd.read_excel('data.xlsx')
print(df.head())

# Specify sheet name
df = pd.read_excel('data.xlsx', sheet_name='Sheet2')

# Read multiple sheets
all_sheets = pd.read_excel('data.xlsx', sheet_name=None)  # Returns a dict of DataFrames
```

### Customizing Excel Import

```python
# Specify cell range
df = pd.read_excel('data.xlsx', usecols='A:C', skiprows=2, nrows=10)

# Read specific columns
df = pd.read_excel('data.xlsx', usecols=[0, 1, 3])  # Columns A, B, and D

# Handle dates
df = pd.read_excel('data.xlsx', parse_dates=['Date'])
```

## Connecting to SQL Databases

Pandas can connect to SQL databases and execute queries using the `read_sql()` function.

### Setting Up Database Connection

First, you need to establish a connection to your database:

```python
import pandas as pd
import sqlite3  # For SQLite
# For other databases: import mysql.connector, import psycopg2, etc.

# Connect to SQLite database
conn = sqlite3.connect('database.db')

# For MySQL
# from sqlalchemy import create_engine
# engine = create_engine('mysql+pymysql://username:password@host/database')
```

### Reading Data from SQL

```python
# Read entire table
df = pd.read_sql('SELECT * FROM employees', conn)

# Read with a WHERE clause
df = pd.read_sql('SELECT * FROM employees WHERE department = "Sales"', conn)

# Parameterized query (safer for user inputs)
dept = 'Sales'
df = pd.read_sql('SELECT * FROM employees WHERE department = ?', conn, params=(dept,))
```

### Using SQLAlchemy for More Complex Database Interactions

```python
from sqlalchemy import create_engine, text

# Create engine
engine = create_engine('sqlite:///database.db')

# Read data
df = pd.read_sql('employees', engine)  # Read entire table
df = pd.read_sql_query(text('SELECT * FROM employees WHERE hire_date > :date'), 
                      engine, 
                      params={'date': '2020-01-01'})
```

## Parsing JSON Data

JSON (JavaScript Object Notation) is a common format for web APIs and configuration files. Pandas can read JSON data using the `read_json()` function.

### Basic JSON Loading

```python
# From a JSON file
df = pd.read_json('data.json')

# From a JSON string
json_string = '{"Name":["John","Anna"],"Age":[28,34]}'
df = pd.read_json(json_string)
```

### Handling Nested JSON

JSON data is often nested, which requires additional processing:

```python
import json

# For complex nested JSON
with open('nested_data.json', 'r') as f:
    data = json.load(f)
    
# Normalize semi-structured JSON data into a flat table
df = pd.json_normalize(data)

# For deeply nested structures
df = pd.json_normalize(data, 
                      record_path=['records', 'values'],
                      meta=['id', ['user', 'name']])
```

## Working with Web Data

Pandas can directly read tables from HTML web pages using the `read_html()` function.

### Reading HTML Tables

```python
# Read all tables from a webpage
tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_countries_by_population')

# Access the first table
df = tables[0]

# Read tables with specific attributes
df = pd.read_html('page.html', match='Population')  # Only tables containing 'Population'
```

### Reading Data from APIs

For more complex web data, you can use requests to fetch data from APIs:

```python
import requests
import pandas as pd

# Make API request
response = requests.get('https://api.example.com/data')
data = response.json()

# Convert to DataFrame
df = pd.DataFrame(data)
# or
df = pd.json_normalize(data['results'])
```

## Other Data Sources

Pandas can read from many other data sources:

### Parquet Files

Parquet is a columnar storage format that's efficient for analytical queries:

```python
# Read Parquet
df = pd.read_parquet('data.parquet')
```

### HDF5 Store

HDF5 is good for storing large amounts of numerical data:

```python
# Read HDF5
df = pd.read_hdf('data.h5', key='df')
```

### Clipboard

You can even read data from your clipboard:

```python
# Read clipboard (useful for data copied from Excel or websites)
df = pd.read_clipboard()
```

## Saving Data

After loading and processing data, you often need to save it:

```python
# Save to CSV
df.to_csv('processed_data.csv', index=False)

# Save to Excel
df.to_excel('processed_data.xlsx', sheet_name='Sheet1')

# Save to SQL
df.to_sql('new_table', conn, if_exists='replace')

# Save to Parquet
df.to_parquet('processed_data.parquet')
```

## Practice Exercises

1. Download a CSV dataset (e.g., from Kaggle) and load it using Pandas.
2. Create an Excel file with multiple sheets and practice reading specific sheets.
3. Set up a SQLite database, create a table, and practice reading from it.
4. Find a public API that returns JSON data, fetch the data, and convert it to a DataFrame.
5. Visit a Wikipedia page with tables and extract a specific table.

## Key Takeaways

- Pandas can read data from various sources: CSV, Excel, SQL, JSON, HTML, and more
- Each reading function has parameters to customize the import process
- For large files, consider reading in chunks to manage memory usage
- Understanding how to load data efficiently is crucial for any data analysis project
- Always check your data after loading to ensure it was imported correctly
