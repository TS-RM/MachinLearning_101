# Beginner-Friendly Data Analysis Projects

## Introduction to Data Analysis Projects

Practical projects are essential for solidifying your data handling and visualization skills. This guide provides a series of beginner-friendly data analysis projects that will help you apply what you've learned about Pandas, data visualization, and exploratory data analysis. Each project includes a description, learning objectives, step-by-step instructions, and tips for extending your analysis.

## Project 1: Analyzing the Iris Dataset

The Iris dataset is a classic dataset for beginners, containing measurements of iris flowers with three different species.

### Learning Objectives
- Load and explore a structured dataset
- Perform basic data cleaning and preprocessing
- Create visualizations to understand the data
- Apply descriptive statistics to summarize findings

### Project Steps

#### Step 1: Load and Explore the Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for visualizations
sns.set_style('whitegrid')

# Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display the first few rows
print("First 5 rows of the Iris dataset:")
print(iris_df.head())

# Check the shape of the dataset
print(f"\nDataset shape: {iris_df.shape}")

# Check for missing values
print("\nMissing values:")
print(iris_df.isnull().sum())

# Get basic information about the dataset
print("\nDataset info:")
print(iris_df.info())

# Summary statistics
print("\nSummary statistics:")
print(iris_df.describe())

# Count of each species
print("\nSpecies distribution:")
print(iris_df['species'].value_counts())
```

#### Step 2: Visualize the Data

```python
# Create histograms for each feature
plt.figure(figsize=(12, 10))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i+1)
    sns.histplot(data=iris_df, x=feature, hue='species', kde=True, bins=20)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# Create a scatter plot matrix
sns.pairplot(iris_df, hue='species', diag_kind='kde')
plt.suptitle('Scatter Plot Matrix of Iris Dataset', y=1.02)
plt.show()

# Create box plots for each feature by species
plt.figure(figsize=(12, 10))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='species', y=feature, data=iris_df)
    plt.title(f'{feature} by Species')
plt.tight_layout()
plt.show()

# Create a violin plot
plt.figure(figsize=(12, 8))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i+1)
    sns.violinplot(x='species', y=feature, data=iris_df, inner='quartile')
    plt.title(f'{feature} by Species')
plt.tight_layout()
plt.show()
```

#### Step 3: Analyze Relationships Between Features

```python
# Calculate correlation matrix
correlation_matrix = iris_df.drop(columns=['species']).corr()

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Iris Features')
plt.tight_layout()
plt.show()

# Analyze correlation by species
species_names = iris_df['species'].unique()
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, species in enumerate(species_names):
    subset = iris_df[iris_df['species'] == species].drop(columns=['species'])
    corr = subset.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[i])
    axes[i].set_title(f'Correlation Matrix for {species}')

plt.tight_layout()
plt.show()
```

#### Step 4: Perform Statistical Analysis

```python
# Calculate descriptive statistics by species
stats_by_species = iris_df.groupby('species').describe()
print("Descriptive statistics by species:")
print(stats_by_species)

# Perform ANOVA to compare means across species
from scipy import stats

for feature in iris.feature_names:
    # Create groups for ANOVA
    groups = [iris_df[iris_df['species'] == species][feature] for species in species_names]
    
    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    print(f"\nANOVA for {feature}:")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
```

#### Step 5: Draw Conclusions

```python
# Create a summary visualization
plt.figure(figsize=(12, 6))

# Plot sepal length vs. sepal width
plt.subplot(1, 2, 1)
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', 
                hue='species', style='species', s=100, data=iris_df)
plt.title('Sepal Length vs. Sepal Width')

# Plot petal length vs. petal width
plt.subplot(1, 2, 2)
sns.scatterplot(x='petal length (cm)', y='petal width (cm)', 
                hue='species', style='species', s=100, data=iris_df)
plt.title('Petal Length vs. Petal Width')

plt.tight_layout()
plt.show()

# Print key findings
print("\nKey Findings from Iris Dataset Analysis:")
print("1. Iris setosa is clearly separable from the other two species based on petal measurements")
print("2. Iris virginica and Iris versicolor have some overlap but can be distinguished")
print("3. Petal length and petal width show the strongest correlation")
print("4. Petal measurements are more useful than sepal measurements for species identification")
print("5. All features show statistically significant differences across species")
```

### Project Extensions
- Apply clustering algorithms (e.g., K-means) to see if they can identify the three species
- Build a simple classification model to predict species based on measurements
- Create an interactive dashboard using tools like Plotly or Panel
- Perform dimensionality reduction using PCA and visualize the results

## Project 2: Creating a Data Cleaning Pipeline

This project focuses on cleaning a messy dataset, a crucial skill for any data analyst.

### Learning Objectives
- Identify and handle missing values
- Detect and address outliers
- Fix inconsistent data formats
- Create a reusable data cleaning pipeline

### Project Steps

#### Step 1: Load and Explore the Messy Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for visualizations
sns.set_style('whitegrid')

# Create a messy dataset for demonstration
np.random.seed(42)
n = 1000

# Create a DataFrame with various issues
messy_data = pd.DataFrame({
    'ID': range(1, n+1),
    'Name': [f'Person_{i}' for i in range(1, n+1)],
    'Age': np.random.randint(18, 80, n),
    'Income': np.random.normal(50000, 15000, n),
    'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD', None], n),
    'Join_Date': pd.date_range(start='2020-01-01', periods=n),
    'Last_Purchase': pd.date_range(start='2020-02-01', periods=n),
    'Customer_Type': np.random.choice(['Regular', 'Premium', 'VIP', 'regular', 'premium', 'vip'], n),
    'Email': [f'person_{i}@example.com' if i % 10 != 0 else None for i in range(1, n+1)]
})

# Introduce missing values
messy_data.loc[np.random.choice(n, 50, replace=False), 'Age'] = np.nan
messy_data.loc[np.random.choice(n, 100, replace=False), 'Income'] = np.nan

# Introduce outliers
messy_data.loc[np.random.choice(n, 10, replace=False), 'Age'] = np.random.randint(100, 120, 10)
messy_data.loc[np.random.choice(n, 10, replace=False), 'Income'] = np.random.uniform(200000, 500000, 10)

# Introduce inconsistent date formats
for i in np.random.choice(n, 50, replace=False):
    messy_data.at[i, 'Join_Date'] = messy_data.at[i, 'Join_Date'].strftime('%m/%d/%Y')
    
for i in np.random.choice(n, 50, replace=False):
    messy_data.at[i, 'Last_Purchase'] = messy_data.at[i, 'Last_Purchase'].strftime('%Y-%m-%d')

# Display the first few rows
print("First 5 rows of the messy dataset:")
print(messy_data.head())

# Check the shape of the dataset
print(f"\nDataset shape: {messy_data.shape}")

# Check for missing values
print("\nMissing values:")
print(messy_data.isnull().sum())

# Get basic information about the dataset
print("\nDataset info:")
print(messy_data.info())

# Summary statistics
print("\nSummary statistics:")
print(messy_data.describe())
```

#### Step 2: Handle Missing Values

```python
# Create a copy of the messy data for cleaning
clean_data = messy_data.copy()

# Check missing values
plt.figure(figsize=(10, 6))
sns.heatmap(clean_data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in Dataset')
plt.tight_layout()
plt.show()

# Handle missing values in Age
# Fill with median
clean_data['Age'].fillna(clean_data['Age'].median(), inplace=True)

# Handle missing values in Income
# Fill with mean
clean_data['Income'].fillna(clean_data['Income'].mean(), inplace=True)

# Handle missing values in Education
# Fill with mode
clean_data['Education'].fillna(clean_data['Education'].mode()[0], inplace=True)

# Handle missing values in Email
# Create a placeholder email based on Name
for idx, row in clean_data[clean_data['Email'].isnull()].iterrows():
    clean_data.at[idx, 'Email'] = f"{row['Name'].lower().replace(' ', '_')}@placeholder.com"

# Verify missing values have been handled
print("\nMissing values after handling:")
print(clean_data.isnull().sum())
```

#### Step 3: Fix Data Types and Formats

```python
# Convert Join_Date to datetime
clean_data['Join_Date'] = pd.to_datetime(clean_data['Join_Date'], errors='coerce')

# Convert Last_Purchase to datetime
clean_data['Last_Purchase'] = pd.to_datetime(clean_data['Last_Purchase'], errors='coerce')

# Handle any remaining date parsing issues
clean_data['Join_Date'].fillna(pd.to_datetime('2020-01-01'), inplace=True)
clean_data['Last_Purchase'].fillna(pd.to_datetime('2020-02-01'), inplace=True)

# Standardize Customer_Type (convert to lowercase and remove spaces)
clean_data['Customer_Type'] = clean_data['Customer_Type'].str.lower().str.strip()

# Verify data types
print("\nData types after fixing:")
print(clean_data.dtypes)
```

#### Step 4: Detect and Handle Outliers

```python
# Visualize outliers in Age and Income
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(y=clean_data['Age'])
plt.title('Boxplot of Age')

plt.subplot(1, 2, 2)
sns.boxplot(y=clean_data['Income'])
plt.title('Boxplot of Income')

plt.tight_layout()
plt.show()

# Define a function to detect outliers using IQR
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Detect outliers in Age
age_outliers, age_lower, age_upper = detect_outliers(clean_data, 'Age')
print(f"\nNumber of outliers in Age: {len(age_outliers)}")
print(f"Age bounds: [{age_lower:.2f}, {age_upper:.2f}]")

# Detect outliers in Income
income_outliers, income_lower, income_upper = detect_outliers(clean_data, 'Income')
print(f"Number of outliers in Income: {len(income_outliers)}")
print(f"Income bounds: [{income_lower:.2f}, {income_upper:.2f}]")

# Handle outliers by capping
clean_data['Age'] = clean_data['Age'].clip(lower=age_lower, upper=age_upper)
clean_data['Income'] = clean_data['Income'].clip(lower=income_lower, upper=income_upper)

# Verify outliers have been handled
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(y=clean_data['Age'])
plt.title('Boxplot of Age (After Handling Outliers)')

plt.subplot(1, 2, 2)
sns.boxplot(y=clean_data['Income'])
plt.title('Boxplot of Income (After Handling Outliers)')

plt.tight_layout()
plt.show()
```

#### Step 5: Create Derived Features

```python
# Calculate days since joining
clean_data['Days_Since_Joining'] = (pd.Timestamp('now') - clean_data['Join_Date']).dt.days

# Calculate days between joining and last purchase
clean_data['Days_To_Purchase'] = (clean_data['Last_Purchase'] - clean_data['Join_Date']).dt.days

# Create age groups
bins = [0, 25, 35, 45, 55, 65, 100]
labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
clean_data['Age_Group'] = pd.cut(clean_data['Age'], bins=bins, labels=labels, right=False)

# Create income groups
income_bins = [0, 30000, 60000, 90000, 120000, float('inf')]
income_labels = ['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High']
clean_data['Income_Group'] = pd.cut(clean_data['Income'], bins=income_bins, labels=income_labels)

# Display the first few rows with new features
print("\nFirst 5 rows with derived features:")
print(clean_data.head())
```

#### Step 6: Create a Reusable Cleaning Pipeline

```python
def clean_customer_data(df):
    """
    A reusable function to clean customer data.
    
    Parameters:
    df (pandas.DataFrame): The messy dataframe to clean
    
    Returns:
    pandas.DataFrame: The cleaned dataframe
    """
    # Create a copy to avoid modifying the original
    clean_df = df.copy()
    
    # 1. Handle missing values
    # Age: Fill with median
    clean_df['Age'].fillna(clean_df['Age'].median(), inplace=True)
    
    # Income: Fill with mean
    clean_df['Income'].fillna(clean_df['Income'].mean(), inplace=True)
    
    # Education: Fill with mode
    if 'Education' in clean_df.columns:
        clean_df['Education'].fillna(clean_df['Education'].mode()[0], inplace=True)
    
    # Email: Create placeholder
    if 'Email' in clean_df.columns and 'Name' in clean_df.columns:
        for idx, row in clean_df[clean_df['Email'].isnull()].iterrows():
            clean_df.at[idx, 'Email'] = f"{row['Name'].lower().replace(' ', '_')}@placeholder.com"
    
    # 2. Fix data types and formats
    # Convert date columns to datetime
    date_columns = ['Join_Date', 'Last_Purchase']
    for col in date_columns:
        if col in clean_df.columns:
            clean_df[col] = pd.to_datetime(clean_df[col], errors='coerce')
            # Fill missing dates with a default
            clean_df[col].fillna(pd.to_datetime('2020-01-01'), inplace=True)
    
    # Standardize categorical columns
    if 'Customer_Type' in clean_df.columns:
        clean_df['Customer_Type'] = clean_df['Customer_Type'].str.lower().str.strip()
    
    # 3. Handle outliers
    numeric_columns = ['Age', 'Income']
    for col in numeric_columns:
        if col in clean_df.columns:
            # Detect outliers
            Q1 = clean_df[col].quantile(0.25)
            Q3 = clean_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            clean_df[col] = clean_df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # 4. Create derived features
    if all(col in clean_df.columns for col in ['Join_Date', 'Last_Purchase']):
        # Days since joining
        clean_df['Days_Since_Joining'] = (pd.Timestamp('now') - clean_df['Join_Date']).dt.days
        
        # Days between joining and last purchase
        clean_df['Days_To_Purchase'] = (clean_df['Last_Purchase'] - clean_df['Join_Date']).dt.days
    
    if 'Age' in clean_df.columns:
        # Age groups
        bins = [0, 25, 35, 45, 55, 65, 100]
        labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        clean_df['Age_Group'] = pd.cut(clean_df['Age'], bins=bins, labels=labels, right=False)
    
    if 'Income' in clean_df.columns:
        # Income groups
        income_bins = [0, 30000, 60000, 90000, 120000, float('inf')]
        income_labels = ['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High']
        clean_df['Income_Group'] = pd.cut(clean_df['Income'], bins=income_bins, labels=income_labels)
    
    return clean_df

# Test the pipeline on our messy data
cleaned_data = clean_customer_data(messy_data)

# Verify the results
print("\nCleaned data info:")
print(cleaned_data.info())
print("\nCleaned data summary statistics:")
print(cleaned_data.describe())

# Check for missing values in the cleaned data
print("\nMissing values in cleaned data:")
print(cleaned_data.isnull().sum())
```

#### Step 7: Analyze the Cleaned Data

```python
# Visualize the distribution of age groups
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(x='Age_Group', data=cleaned_data)
plt.title('Distribution of Age Groups')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.countplot(x='Income_Group', data=cleaned_data)
plt.title('Distribution of Income Groups')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Analyze customer types
plt.figure(figsize=(10, 6))
sns.countplot(x='Customer_Type', data=cleaned_data)
plt.title('Distribution of Customer Types')
plt.tight_layout()
plt.show()

# Analyze days to purchase by customer type
plt.figure(figsize=(10, 6))
sns.boxplot(x='Customer_Type', y='Days_To_Purchase', data=cleaned_data)
plt.title('Days to Purchase by Customer Type')
plt.tight_layout()
plt.show()

# Analyze income by education level
plt.figure(figsize=(10, 6))
sns.boxplot(x='Education', y='Income', data=cleaned_data)
plt.title('Income by Education Level')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Project Extensions
- Apply the cleaning pipeline to a real-world messy dataset
- Add data validation rules to ensure data quality
- Create a more sophisticated outlier detection method
- Build a data quality dashboard to monitor data issues
- Implement the pipeline as a reusable Python package

## Project 3: Building a Dashboard with Basic Visualizations

This project focuses on creating a comprehensive dashboard to visualize and explore data.

### Learning Objectives
- Design an effective dashboard layout
- Create various types of visualizations
- Combine multiple visualizations to tell a data story
- Practice data aggregation and transformation

### Project Steps

#### Step 1: Load and Prepare the Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

# Load a sample dataset (sales data)
# For this example, we'll create synthetic sales data
np.random.seed(42)

# Create date range for one year of daily data
dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
n_dates = len(dates)

# Create product categories and regions
products = ['Electronics', 'Clothing', 'Home', 'Sports', 'Books']
regions = ['North', 'South', 'East', 'West', 'Central']

# Generate sales data
sales_data = []
for date in dates:
    # Add seasonality effect
    season_factor = 1.0
    if date.month in [11, 12]:  # Holiday season
        season_factor = 1.5
    elif date.month in [6, 7, 8]:  # Summer
        season_factor = 1.2
    
    # Weekend effect
    weekend_factor = 1.3 if date.dayofweek >= 5 else 1.0
    
    # Generate multiple sales records for each day
    n_records = np.random.randint(50, 100)
    for _ in range(n_records):
        product = np.random.choice(products)
        region = np.random.choice(regions)
        
        # Base price and quantity vary by product
        if product == 'Electronics':
            base_price = np.random.uniform(100, 1000)
            base_quantity = np.random.randint(1, 3)
        elif product == 'Clothing':
            base_price = np.random.uniform(20, 200)
            base_quantity = np.random.randint(1, 5)
        elif product == 'Home':
            base_price = np.random.uniform(50, 500)
            base_quantity = np.random.randint(1, 4)
        elif product == 'Sports':
            base_price = np.random.uniform(30, 300)
            base_quantity = np.random.randint(1, 3)
        else:  # Books
            base_price = np.random.uniform(10, 50)
            base_quantity = np.random.randint(1, 6)
        
        # Apply factors
        quantity = max(1, int(base_quantity * weekend_factor * season_factor * np.random.uniform(0.8, 1.2)))
        price = base_price * np.random.uniform(0.9, 1.1)
        
        # Calculate total
        total = price * quantity
        
        # Add to sales data
        sales_data.append({
            'Date': date,
            'Product': product,
            'Region': region,
            'Quantity': quantity,
            'Price': price,
            'Total': total
        })

# Create DataFrame
sales_df = pd.DataFrame(sales_data)

# Add some derived columns
sales_df['Month'] = sales_df['Date'].dt.month_name()
sales_df['Day'] = sales_df['Date'].dt.day_name()
sales_df['Week'] = sales_df['Date'].dt.isocalendar().week
sales_df['Quarter'] = sales_df['Date'].dt.quarter

# Display the first few rows
print("First 5 rows of the sales dataset:")
print(sales_df.head())

# Check the shape of the dataset
print(f"\nDataset shape: {sales_df.shape}")

# Summary statistics
print("\nSummary statistics:")
print(sales_df.describe())
```

#### Step 2: Create Time Series Visualizations

```python
# Aggregate sales by date
daily_sales = sales_df.groupby('Date')['Total'].sum().reset_index()
monthly_sales = sales_df.groupby('Month')['Total'].sum().reset_index()

# Reorder months
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_sales['Month'] = pd.Categorical(monthly_sales['Month'], categories=month_order, ordered=True)
monthly_sales = monthly_sales.sort_values('Month')

# Create time series plots
plt.figure(figsize=(15, 10))

# Daily sales trend
plt.subplot(2, 1, 1)
plt.plot(daily_sales['Date'], daily_sales['Total'], marker='', linewidth=1)
plt.title('Daily Sales Trend (2022)')
plt.xlabel('Date')
plt.ylabel('Total Sales ($)')
plt.grid(True, alpha=0.3)

# Monthly sales trend
plt.subplot(2, 1, 2)
sns.barplot(x='Month', y='Total', data=monthly_sales)
plt.title('Monthly Sales (2022)')
plt.xlabel('Month')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Sales by day of week
day_sales = sales_df.groupby('Day')['Total'].sum().reset_index()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_sales['Day'] = pd.Categorical(day_sales['Day'], categories=day_order, ordered=True)
day_sales = day_sales.sort_values('Day')

plt.figure(figsize=(12, 6))
sns.barplot(x='Day', y='Total', data=day_sales)
plt.title('Sales by Day of Week')
plt.xlabel('Day')
plt.ylabel('Total Sales ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Quarterly sales
quarterly_sales = sales_df.groupby('Quarter')['Total'].sum().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Quarter', y='Total', data=quarterly_sales)
plt.title('Quarterly Sales (2022)')
plt.xlabel('Quarter')
plt.ylabel('Total Sales ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

#### Step 3: Create Category Comparison Visualizations

```python
# Sales by product category
product_sales = sales_df.groupby('Product')['Total'].sum().reset_index()
product_sales = product_sales.sort_values('Total', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Product', y='Total', data=product_sales)
plt.title('Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Sales ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Sales by region
region_sales = sales_df.groupby('Region')['Total'].sum().reset_index()
region_sales = region_sales.sort_values('Total', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Region', y='Total', data=region_sales)
plt.title('Sales by Region')
plt.xlabel('Region')
plt.ylabel('Total Sales ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Product category by region
product_region_sales = sales_df.groupby(['Region', 'Product'])['Total'].sum().reset_index()

plt.figure(figsize=(14, 8))
sns.barplot(x='Region', y='Total', hue='Product', data=product_region_sales)
plt.title('Sales by Region and Product Category')
plt.xlabel('Region')
plt.ylabel('Total Sales ($)')
plt.legend(title='Product Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

#### Step 4: Create Distribution Visualizations

```python
# Distribution of sales amounts
plt.figure(figsize=(12, 6))
sns.histplot(sales_df['Total'], bins=50, kde=True)
plt.title('Distribution of Sales Amounts')
plt.xlabel('Sale Amount ($)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Distribution of quantities
plt.figure(figsize=(12, 6))
sns.countplot(x='Quantity', data=sales_df)
plt.title('Distribution of Quantities Sold')
plt.xlabel('Quantity')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Distribution of prices by product category
plt.figure(figsize=(14, 8))
sns.boxplot(x='Product', y='Price', data=sales_df)
plt.title('Price Distribution by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Price ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

#### Step 5: Create Relationship Visualizations

```python
# Relationship between price and quantity
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Price', y='Quantity', hue='Product', size='Total', 
                sizes=(20, 200), alpha=0.7, data=sales_df.sample(1000))
plt.title('Relationship Between Price and Quantity')
plt.xlabel('Price ($)')
plt.ylabel('Quantity')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Heatmap of sales by month and day of week
# First, create a pivot table
month_day_sales = sales_df.groupby(['Month', 'Day'])['Total'].sum().reset_index()
month_day_pivot = month_day_sales.pivot(index='Day', columns='Month', values='Total')

# Reorder rows and columns
month_day_pivot = month_day_pivot.reindex(day_order)
month_day_pivot = month_day_pivot[month_order]

plt.figure(figsize=(14, 8))
sns.heatmap(month_day_pivot, annot=False, cmap='YlGnBu', fmt='.0f')
plt.title('Sales Heatmap by Month and Day of Week')
plt.tight_layout()
plt.show()
```

#### Step 6: Create a Comprehensive Dashboard

```python
# Create a comprehensive dashboard with multiple visualizations
plt.figure(figsize=(20, 15))

# Monthly sales trend
plt.subplot(3, 2, 1)
sns.barplot(x='Month', y='Total', data=monthly_sales)
plt.title('Monthly Sales (2022)')
plt.xlabel('Month')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Sales by product category
plt.subplot(3, 2, 2)
sns.barplot(x='Product', y='Total', data=product_sales)
plt.title('Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Sales ($)')
plt.grid(True, alpha=0.3)

# Sales by region
plt.subplot(3, 2, 3)
sns.barplot(x='Region', y='Total', data=region_sales)
plt.title('Sales by Region')
plt.xlabel('Region')
plt.ylabel('Total Sales ($)')
plt.grid(True, alpha=0.3)

# Sales by day of week
plt.subplot(3, 2, 4)
sns.barplot(x='Day', y='Total', data=day_sales)
plt.title('Sales by Day of Week')
plt.xlabel('Day')
plt.ylabel('Total Sales ($)')
plt.grid(True, alpha=0.3)

# Distribution of sales amounts
plt.subplot(3, 2, 5)
sns.histplot(sales_df['Total'], bins=50, kde=True)
plt.title('Distribution of Sales Amounts')
plt.xlabel('Sale Amount ($)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Price distribution by product
plt.subplot(3, 2, 6)
sns.boxplot(x='Product', y='Price', data=sales_df)
plt.title('Price Distribution by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Price ($)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sales_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Step 7: Create an Interactive Dashboard (Optional)

```python
# Note: This requires additional libraries like Plotly or Panel
# Here's an example using Plotly Express

# !pip install plotly

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Monthly sales trend
fig1 = px.bar(monthly_sales, x='Month', y='Total', 
             title='Monthly Sales (2022)')
fig1.update_xaxes(categoryorder='array', categoryarray=month_order)

# Sales by product category
fig2 = px.bar(product_sales, x='Product', y='Total', 
             title='Sales by Product Category')

# Sales by region
fig3 = px.bar(region_sales, x='Region', y='Total', 
             title='Sales by Region')

# Create a dashboard with subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Monthly Sales (2022)', 'Sales by Product Category', 
                   'Sales by Region', 'Sales by Day of Week')
)

# Add traces
for i, month in enumerate(monthly_sales['Month']):
    fig.add_trace(
        go.Bar(x=[month], y=[monthly_sales.iloc[i]['Total']], name=month, showlegend=False),
        row=1, col=1
    )

for i, product in enumerate(product_sales['Product']):
    fig.add_trace(
        go.Bar(x=[product], y=[product_sales.iloc[i]['Total']], name=product, showlegend=False),
        row=1, col=2
    )

for i, region in enumerate(region_sales['Region']):
    fig.add_trace(
        go.Bar(x=[region], y=[region_sales.iloc[i]['Total']], name=region, showlegend=False),
        row=2, col=1
    )

for i, day in enumerate(day_sales['Day']):
    fig.add_trace(
        go.Bar(x=[day], y=[day_sales.iloc[i]['Total']], name=day, showlegend=False),
        row=2, col=2
    )

# Update layout
fig.update_layout(height=800, width=1200, title_text="Sales Dashboard")
fig.show()

# Save the interactive dashboard
fig.write_html('interactive_sales_dashboard.html')
```

### Project Extensions
- Add interactive filters to the dashboard
- Create a geographic visualization of sales by region
- Add trend analysis and forecasting
- Build a real-time dashboard that updates automatically
- Create a dashboard with user controls for different time periods

## Project 4: Performing EDA on a Public Health Dataset

This project focuses on exploratory data analysis of a public health dataset.

### Learning Objectives
- Apply EDA techniques to a real-world dataset
- Identify patterns and trends in health data
- Create informative visualizations for health metrics
- Draw meaningful conclusions from the analysis

### Project Steps

#### Step 1: Load and Explore the Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for visualizations
sns.set_style('whitegrid')

# Load a public health dataset (diabetes dataset)
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target, name='disease_progression')

# Combine features and target
df = pd.concat([X, y], axis=1)

# Display the first few rows
print("First 5 rows of the diabetes dataset:")
print(df.head())

# Check the shape of the dataset
print(f"\nDataset shape: {df.shape}")

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Get basic information about the dataset
print("\nDataset info:")
print(df.info())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Feature descriptions (from the dataset documentation)
feature_descriptions = {
    'age': 'Age',
    'sex': 'Sex',
    'bmi': 'Body Mass Index',
    'bp': 'Average Blood Pressure',
    's1': 'Total Serum Cholesterol',
    's2': 'Low-Density Lipoproteins',
    's3': 'High-Density Lipoproteins',
    's4': 'Total Cholesterol / HDL',
    's5': 'Log of Serum Triglycerides Level',
    's6': 'Blood Sugar Level',
    'disease_progression': 'Disease Progression Indicator'
}

print("\nFeature descriptions:")
for feature, description in feature_descriptions.items():
    print(f"{feature}: {description}")
```

#### Step 2: Analyze Distributions of Health Metrics

```python
# Create histograms for all features
plt.figure(figsize=(15, 12))
for i, feature in enumerate(df.columns):
    plt.subplot(4, 3, i+1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature_descriptions.get(feature, feature)}')
    plt.xlabel(feature)
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Create box plots for all features
plt.figure(figsize=(15, 10))
df.boxplot(figsize=(15, 10))
plt.title('Box Plots of Health Metrics')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Check for skewness in the distributions
skewness = df.skew()
print("\nSkewness of distributions:")
for feature, skew_value in skewness.items():
    print(f"{feature}: {skew_value:.4f}")

# Visualize the most skewed features
most_skewed = skewness.abs().sort_values(ascending=False).index[:3]
plt.figure(figsize=(15, 5))
for i, feature in enumerate(most_skewed):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature_descriptions.get(feature, feature)}\nSkewness: {skewness[feature]:.4f}')
    plt.xlabel(feature)
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

#### Step 3: Analyze Relationships Between Health Metrics

```python
# Calculate correlation matrix
correlation_matrix = df.corr()

# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Health Metrics')
plt.tight_layout()
plt.show()

# Identify the most correlated features with disease progression
target_correlations = correlation_matrix['disease_progression'].drop('disease_progression').abs().sort_values(ascending=False)
print("\nFeatures most correlated with disease progression:")
for feature, corr in target_correlations.items():
    print(f"{feature_descriptions.get(feature, feature)}: {corr:.4f}")

# Visualize the top 3 most correlated features with disease progression
top_features = target_correlations.index[:3]
plt.figure(figsize=(15, 5))
for i, feature in enumerate(top_features):
    plt.subplot(1, 3, i+1)
    sns.scatterplot(x=feature, y='disease_progression', data=df)
    plt.title(f'{feature_descriptions.get(feature, feature)} vs. Disease Progression\nCorrelation: {correlation_matrix.loc[feature, "disease_progression"]:.4f}')
    plt.xlabel(feature_descriptions.get(feature, feature))
    plt.ylabel('Disease Progression')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Create a pair plot for the most important features
important_features = list(top_features) + ['disease_progression']
sns.pairplot(df[important_features])
plt.suptitle('Pair Plot of Key Health Metrics', y=1.02)
plt.tight_layout()
plt.show()
```

#### Step 4: Analyze Group Differences

```python
# Create a binary sex variable for easier interpretation (assuming 0 = female, 1 = male in the original data)
df['sex_category'] = df['sex'].apply(lambda x: 'Male' if x > 0 else 'Female')

# Compare health metrics by sex
plt.figure(figsize=(15, 10))
for i, feature in enumerate(df.columns[:-2]):  # Exclude sex_category and original sex
    if feature != 'sex':
        plt.subplot(3, 3, i)
        sns.boxplot(x='sex_category', y=feature, data=df)
        plt.title(f'{feature_descriptions.get(feature, feature)} by Sex')
        plt.xlabel('Sex')
        plt.ylabel(feature)
        plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Create BMI categories
bmi_bins = [0, 18.5, 25, 30, 100]
bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
df['bmi_category'] = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels)

# Compare disease progression by BMI category
plt.figure(figsize=(10, 6))
sns.boxplot(x='bmi_category', y='disease_progression', data=df)
plt.title('Disease Progression by BMI Category')
plt.xlabel('BMI Category')
plt.ylabel('Disease Progression')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Create blood pressure categories
bp_bins = [0, 80, 120, 140, 200]
bp_labels = ['Low', 'Normal', 'Elevated', 'High']
df['bp_category'] = pd.cut(df['bp'], bins=bp_bins, labels=bp_labels)

# Compare disease progression by blood pressure category
plt.figure(figsize=(10, 6))
sns.boxplot(x='bp_category', y='disease_progression', data=df)
plt.title('Disease Progression by Blood Pressure Category')
plt.xlabel('Blood Pressure Category')
plt.ylabel('Disease Progression')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

#### Step 5: Perform Statistical Tests

```python
from scipy import stats

# Compare disease progression between males and females
male_progression = df[df['sex_category'] == 'Male']['disease_progression']
female_progression = df[df['sex_category'] == 'Female']['disease_progression']

t_stat, p_value = stats.ttest_ind(male_progression, female_progression)
print(f"\nT-test for disease progression between males and females:")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

# ANOVA for disease progression across BMI categories
groups = [df[df['bmi_category'] == cat]['disease_progression'] for cat in df['bmi_category'].unique()]
f_stat, p_value = stats.f_oneway(*groups)
print(f"\nANOVA for disease progression across BMI categories:")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

# Correlation test for BMI and disease progression
corr, p_value = stats.pearsonr(df['bmi'], df['disease_progression'])
print(f"\nCorrelation test for BMI and disease progression:")
print(f"Correlation coefficient: {corr:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Significant correlation: {'Yes' if p_value < 0.05 else 'No'}")
```

#### Step 6: Create a Multivariate Analysis

```python
# Create a 3D scatter plot of the three most important features
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Get the top 3 features
x = df[top_features[0]]
y = df[top_features[1]]
z = df[top_features[2]]
colors = df['disease_progression']

scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=50, alpha=0.8)
ax.set_xlabel(feature_descriptions.get(top_features[0], top_features[0]))
ax.set_ylabel(feature_descriptions.get(top_features[1], top_features[1]))
ax.set_zlabel(feature_descriptions.get(top_features[2], top_features[2]))
plt.colorbar(scatter, label='Disease Progression')
plt.title('3D Scatter Plot of Key Health Metrics')
plt.tight_layout()
plt.show()

# Create a heatmap of disease progression by BMI and blood pressure
pivot_table = df.pivot_table(values='disease_progression', 
                             index='bmi_category', 
                             columns='bp_category', 
                             aggfunc='mean')

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', fmt='.1f')
plt.title('Average Disease Progression by BMI and Blood Pressure')
plt.tight_layout()
plt.show()
```

#### Step 7: Draw Conclusions and Create a Summary Report

```python
# Create a summary visualization
plt.figure(figsize=(15, 10))

# Top correlations with disease progression
plt.subplot(2, 2, 1)
sns.barplot(x=target_correlations.index[:5], y=target_correlations.values[:5])
plt.title('Top Correlations with Disease Progression')
plt.xlabel('Health Metric')
plt.ylabel('Absolute Correlation')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Disease progression by sex
plt.subplot(2, 2, 2)
sns.boxplot(x='sex_category', y='disease_progression', data=df)
plt.title('Disease Progression by Sex')
plt.xlabel('Sex')
plt.ylabel('Disease Progression')
plt.grid(True, alpha=0.3)

# Disease progression by BMI category
plt.subplot(2, 2, 3)
sns.boxplot(x='bmi_category', y='disease_progression', data=df)
plt.title('Disease Progression by BMI Category')
plt.xlabel('BMI Category')
plt.ylabel('Disease Progression')
plt.grid(True, alpha=0.3)

# Scatter plot of top predictor vs. disease progression
plt.subplot(2, 2, 4)
top_feature = top_features[0]
sns.regplot(x=top_feature, y='disease_progression', data=df)
plt.title(f'{feature_descriptions.get(top_feature, top_feature)} vs. Disease Progression')
plt.xlabel(feature_descriptions.get(top_feature, top_feature))
plt.ylabel('Disease Progression')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diabetes_analysis_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# Print key findings
print("\nKey Findings from Diabetes Dataset Analysis:")
print("1. The most important predictors of disease progression are BMI, Blood Pressure, and S5 (Log of Serum Triglycerides Level)")
print(f"2. BMI has a correlation of {correlation_matrix.loc['bmi', 'disease_progression']:.4f} with disease progression")
print(f"3. Blood Pressure has a correlation of {correlation_matrix.loc['bp', 'disease_progression']:.4f} with disease progression")
print("4. There are significant differences in disease progression across BMI categories")
print("5. Higher BMI and blood pressure are associated with increased disease progression")
print("6. The relationship between BMI and disease progression is statistically significant")
```

### Project Extensions
- Apply machine learning models to predict disease progression
- Perform more advanced statistical analyses (e.g., multiple regression)
- Create an interactive dashboard for exploring the health data
- Compare this dataset with other public health datasets
- Develop health risk profiles based on the analysis

## Project 5: Creating a Portfolio Project Analyzing Personal Finance Data

This project focuses on analyzing personal finance data to gain insights into spending patterns and financial health.

### Learning Objectives
- Clean and organize financial transaction data
- Identify spending patterns and trends
- Create visualizations for financial analysis
- Develop actionable insights from financial data

### Project Steps

#### Step 1: Create and Load Sample Financial Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set the style for visualizations
sns.set_style('whitegrid')

# Create sample financial data
np.random.seed(42)

# Generate dates for one year
start_date = datetime(2022, 1, 1)
end_date = datetime(2022, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Create categories and subcategories
categories = {
    'Income': ['Salary', 'Bonus', 'Interest', 'Other Income'],
    'Housing': ['Rent/Mortgage', 'Utilities', 'Maintenance', 'Insurance'],
    'Food': ['Groceries', 'Restaurants', 'Takeout'],
    'Transportation': ['Gas', 'Public Transit', 'Car Maintenance', 'Rideshare'],
    'Entertainment': ['Movies', 'Subscriptions', 'Hobbies', 'Travel'],
    'Shopping': ['Clothing', 'Electronics', 'Home Goods'],
    'Health': ['Medical', 'Pharmacy', 'Fitness'],
    'Education': ['Tuition', 'Books', 'Courses'],
    'Personal': ['Grooming', 'Gifts', 'Charity'],
    'Savings': ['Emergency Fund', 'Retirement', 'Investments']
}

# Generate transactions
transactions = []

# Add regular income (bi-weekly salary)
for pay_date in pd.date_range(start=start_date, end=end_date, freq='2W'):
    salary = np.random.normal(3000, 100)  # Bi-weekly salary around $3000
    transactions.append({
        'Date': pay_date,
        'Category': 'Income',
        'Subcategory': 'Salary',
        'Amount': salary,
        'Description': 'Bi-weekly salary'
    })

# Add quarterly bonuses
for quarter in range(1, 5):
    bonus_date = datetime(2022, quarter * 3, 15)
    if bonus_date <= end_date:
        bonus = np.random.normal(1500, 300)  # Quarterly bonus around $1500
        transactions.append({
            'Date': bonus_date,
            'Category': 'Income',
            'Subcategory': 'Bonus',
            'Amount': bonus,
            'Description': f'Q{quarter} performance bonus'
        })

# Add monthly expenses
for month in range(1, 13):
    month_start = datetime(2022, month, 1)
    days_in_month = (datetime(2022, month + 1, 1) if month < 12 else datetime(2023, 1, 1)) - timedelta(days=1)
    days_in_month = days_in_month.day
    
    # Rent/Mortgage (monthly)
    rent_date = datetime(2022, month, 1)
    rent = np.random.normal(1500, 50)  # Monthly rent around $1500
    transactions.append({
        'Date': rent_date,
        'Category': 'Housing',
        'Subcategory': 'Rent/Mortgage',
        'Amount': -rent,  # Negative for expenses
        'Description': 'Monthly rent'
    })
    
    # Utilities (monthly)
    utilities_date = datetime(2022, month, 15)
    utilities = np.random.normal(200, 50)  # Monthly utilities around $200
    transactions.append({
        'Date': utilities_date,
        'Category': 'Housing',
        'Subcategory': 'Utilities',
        'Amount': -utilities,
        'Description': 'Monthly utilities'
    })
    
    # Groceries (weekly)
    for week in range(4):
        grocery_date = month_start + timedelta(days=week*7 + np.random.randint(0, 7))
        if grocery_date <= end_date:
            groceries = np.random.normal(100, 30)  # Weekly groceries around $100
            transactions.append({
                'Date': grocery_date,
                'Category': 'Food',
                'Subcategory': 'Groceries',
                'Amount': -groceries,
                'Description': 'Weekly groceries'
            })
    
    # Restaurants (random, 5-10 times per month)
    num_restaurant_visits = np.random.randint(5, 11)
    for _ in range(num_restaurant_visits):
        restaurant_date = month_start + timedelta(days=np.random.randint(0, days_in_month))
        if restaurant_date <= end_date:
            restaurant_cost = np.random.normal(50, 20)  # Restaurant meal around $50
            transactions.append({
                'Date': restaurant_date,
                'Category': 'Food',
                'Subcategory': 'Restaurants',
                'Amount': -restaurant_cost,
                'Description': 'Restaurant meal'
            })
    
    # Gas (bi-weekly)
    for bi_week in range(2):
        gas_date = month_start + timedelta(days=bi_week*14 + np.random.randint(0, 7))
        if gas_date <= end_date:
            gas_cost = np.random.normal(40, 10)  # Gas fill-up around $40
            transactions.append({
                'Date': gas_date,
                'Category': 'Transportation',
                'Subcategory': 'Gas',
                'Amount': -gas_cost,
                'Description': 'Gas fill-up'
            })
    
    # Subscriptions (monthly)
    subscription_date = datetime(2022, month, 10)
    subscription_cost = np.random.normal(50, 5)  # Monthly subscriptions around $50
    transactions.append({
        'Date': subscription_date,
        'Category': 'Entertainment',
        'Subcategory': 'Subscriptions',
        'Amount': -subscription_cost,
        'Description': 'Monthly subscriptions'
    })
    
    # Shopping (random, 3-8 times per month)
    num_shopping_trips = np.random.randint(3, 9)
    for _ in range(num_shopping_trips):
        shopping_date = month_start + timedelta(days=np.random.randint(0, days_in_month))
        if shopping_date <= end_date:
            shopping_cost = np.random.normal(60, 40)  # Shopping trip around $60
            subcategory = np.random.choice(categories['Shopping'])
            transactions.append({
                'Date': shopping_date,
                'Category': 'Shopping',
                'Subcategory': subcategory,
                'Amount': -shopping_cost,
                'Description': f'{subcategory} purchase'
            })
    
    # Savings (monthly)
    savings_date = datetime(2022, month, 5)
    savings_amount = np.random.normal(500, 100)  # Monthly savings around $500
    subcategory = np.random.choice(categories['Savings'])
    transactions.append({
        'Date': savings_date,
        'Category': 'Savings',
        'Subcategory': subcategory,
        'Amount': -savings_amount,  # Negative as it's money going out of checking
        'Description': f'Monthly {subcategory.lower()}'
    })

# Add some random transactions for other categories
for _ in range(200):
    random_date = start_date + timedelta(days=np.random.randint(0, 365))
    if random_date <= end_date:
        category = np.random.choice(list(categories.keys()))
        if category not in ['Income', 'Housing', 'Food', 'Transportation', 'Entertainment', 'Shopping', 'Savings']:
            subcategory = np.random.choice(categories[category])
            amount = -np.random.normal(70, 50)  # Random expense around $70
            transactions.append({
                'Date': random_date,
                'Category': category,
                'Subcategory': subcategory,
                'Amount': amount,
                'Description': f'{subcategory} expense'
            })

# Create DataFrame
finance_df = pd.DataFrame(transactions)

# Sort by date
finance_df = finance_df.sort_values('Date')

# Add month and day of week columns
finance_df['Month'] = finance_df['Date'].dt.month_name()
finance_df['Day'] = finance_df['Date'].dt.day_name()
finance_df['Week'] = finance_df['Date'].dt.isocalendar().week

# Display the first few rows
print("First 5 rows of the financial dataset:")
print(finance_df.head())

# Check the shape of the dataset
print(f"\nDataset shape: {finance_df.shape}")

# Summary statistics
print("\nSummary statistics:")
print(finance_df.describe())

# Count by category
print("\nTransaction count by category:")
print(finance_df['Category'].value_counts())
```

#### Step 2: Clean and Prepare the Data

```python
# Check for missing values
print("\nMissing values:")
print(finance_df.isnull().sum())

# Check for duplicate transactions
duplicates = finance_df.duplicated(subset=['Date', 'Category', 'Subcategory', 'Amount'])
print(f"\nNumber of duplicate transactions: {duplicates.sum()}")

# Remove duplicates if any
if duplicates.sum() > 0:
    finance_df = finance_df.drop_duplicates(subset=['Date', 'Category', 'Subcategory', 'Amount'])

# Create a 'Transaction Type' column
finance_df['Transaction Type'] = finance_df['Amount'].apply(lambda x: 'Income' if x > 0 else 'Expense')

# Create an 'Absolute Amount' column for easier analysis
finance_df['Absolute Amount'] = finance_df['Amount'].abs()

# Reorder months for proper chronological display
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
finance_df['Month'] = pd.Categorical(finance_df['Month'], categories=month_order, ordered=True)

# Reorder days for proper weekly display
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
finance_df['Day'] = pd.Categorical(finance_df['Day'], categories=day_order, ordered=True)

# Display the cleaned data
print("\nCleaned data:")
print(finance_df.head())
```

#### Step 3: Analyze Income and Expenses

```python
# Calculate monthly income and expenses
monthly_summary = finance_df.groupby(['Month', 'Transaction Type'])['Amount'].sum().unstack().reset_index()
monthly_summary['Net'] = monthly_summary['Income'] + monthly_summary['Expense']  # Expense is negative

# Ensure all months are included
for month in month_order:
    if month not in monthly_summary['Month'].values:
        monthly_summary = monthly_summary.append({'Month': month, 'Income': 0, 'Expense': 0, 'Net': 0}, ignore_index=True)

# Sort by month
monthly_summary['Month'] = pd.Categorical(monthly_summary['Month'], categories=month_order, ordered=True)
monthly_summary = monthly_summary.sort_values('Month')

# Visualize monthly income and expenses
plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.bar(monthly_summary['Month'], monthly_summary['Income'], color='green', alpha=0.7, label='Income')
plt.bar(monthly_summary['Month'], monthly_summary['Expense'], color='red', alpha=0.7, label='Expenses')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.title('Monthly Income and Expenses')
plt.xlabel('Month')
plt.ylabel('Amount ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Visualize net income
plt.subplot(2, 1, 2)
colors = ['green' if x > 0 else 'red' for x in monthly_summary['Net']]
plt.bar(monthly_summary['Month'], monthly_summary['Net'], color=colors, alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.title('Monthly Net Income')
plt.xlabel('Month')
plt.ylabel('Amount ($)')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Calculate income sources
income_sources = finance_df[finance_df['Transaction Type'] == 'Income'].groupby('Subcategory')['Amount'].sum().sort_values(ascending=False)

# Visualize income sources
plt.figure(figsize=(10, 6))
plt.pie(income_sources, labels=income_sources.index, autopct='%1.1f%%', startangle=90, shadow=True)
plt.axis('equal')
plt.title('Income Sources')
plt.tight_layout()
plt.show()

# Calculate total income, expenses, and savings rate
total_income = finance_df[finance_df['Transaction Type'] == 'Income']['Amount'].sum()
total_expenses = finance_df[finance_df['Transaction Type'] == 'Expense']['Amount'].sum()
savings = finance_df[finance_df['Category'] == 'Savings']['Amount'].abs().sum()
savings_rate = (savings / total_income) * 100

print(f"\nTotal Income: ${total_income:.2f}")
print(f"Total Expenses: ${abs(total_expenses):.2f}")
print(f"Net Income: ${total_income + total_expenses:.2f}")
print(f"Total Savings: ${savings:.2f}")
print(f"Savings Rate: {savings_rate:.2f}%")
```

#### Step 4: Analyze Spending Patterns

```python
# Analyze spending by category
expenses = finance_df[finance_df['Transaction Type'] == 'Expense']
category_expenses = expenses.groupby('Category')['Absolute Amount'].sum().sort_values(ascending=False)

# Visualize spending by category
plt.figure(figsize=(12, 6))
sns.barplot(x=category_expenses.index, y=category_expenses.values)
plt.title('Expenses by Category')
plt.xlabel('Category')
plt.ylabel('Amount ($)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Analyze spending by subcategory (top 15)
subcategory_expenses = expenses.groupby('Subcategory')['Absolute Amount'].sum().sort_values(ascending=False).head(15)

# Visualize spending by subcategory
plt.figure(figsize=(12, 6))
sns.barplot(x=subcategory_expenses.index, y=subcategory_expenses.values)
plt.title('Top 15 Expenses by Subcategory')
plt.xlabel('Subcategory')
plt.ylabel('Amount ($)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Analyze spending by day of week
day_expenses = expenses.groupby('Day')['Absolute Amount'].sum()

# Visualize spending by day of week
plt.figure(figsize=(10, 6))
sns.barplot(x=day_expenses.index, y=day_expenses.values)
plt.title('Expenses by Day of Week')
plt.xlabel('Day')
plt.ylabel('Amount ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Analyze spending trends over time
monthly_category_expenses = expenses.groupby(['Month', 'Category'])['Absolute Amount'].sum().reset_index()

# Visualize spending trends for top 5 categories
top_categories = category_expenses.head(5).index
plt.figure(figsize=(14, 8))
for category in top_categories:
    category_data = monthly_category_expenses[monthly_category_expenses['Category'] == category]
    plt.plot(category_data['Month'], category_data['Absolute Amount'], marker='o', label=category)
plt.title('Monthly Spending Trends by Category')
plt.xlabel('Month')
plt.ylabel('Amount ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

#### Step 5: Perform Budget Analysis

```python
# Create a simple budget based on average monthly expenses
monthly_expenses = expenses.groupby(pd.Grouper(key='Date', freq='M'))['Absolute Amount'].sum()
average_monthly_expense = monthly_expenses.mean()

# Calculate average monthly expenses by category
monthly_category_avg = expenses.groupby(['Category', pd.Grouper(key='Date', freq='M')])['Absolute Amount'].sum().groupby('Category').mean().sort_values(ascending=False)

print("\nAverage Monthly Expenses:")
print(f"Total: ${average_monthly_expense:.2f}")
print("\nBy Category:")
for category, amount in monthly_category_avg.items():
    print(f"{category}: ${amount:.2f}")

# Create a budget vs. actual comparison
budget = {
    'Housing': 1700,
    'Food': 600,
    'Transportation': 200,
    'Entertainment': 300,
    'Shopping': 400,
    'Health': 200,
    'Education': 100,
    'Personal': 200,
    'Savings': 500
}

# Convert budget to DataFrame
budget_df = pd.DataFrame(list(budget.items()), columns=['Category', 'Budget'])
actual_df = pd.DataFrame(monthly_category_avg).reset_index()
actual_df.columns = ['Category', 'Actual']

# Merge budget and actual
budget_comparison = pd.merge(budget_df, actual_df, on='Category', how='outer')
budget_comparison['Difference'] = budget_comparison['Budget'] - budget_comparison['Actual']
budget_comparison['Percent'] = (budget_comparison['Actual'] / budget_comparison['Budget'] * 100).round(1)

print("\nBudget vs. Actual Comparison:")
print(budget_comparison)

# Visualize budget vs. actual
plt.figure(figsize=(12, 6))
x = range(len(budget_comparison))
width = 0.35
plt.bar([i - width/2 for i in x], budget_comparison['Budget'], width, label='Budget', color='blue', alpha=0.7)
plt.bar([i + width/2 for i in x], budget_comparison['Actual'], width, label='Actual', color='orange', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.xticks(x, budget_comparison['Category'], rotation=45)
plt.title('Budget vs. Actual Expenses by Category')
plt.xlabel('Category')
plt.ylabel('Amount ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Visualize budget variance
plt.figure(figsize=(12, 6))
colors = ['green' if x > 0 else 'red' for x in budget_comparison['Difference']]
plt.bar(budget_comparison['Category'], budget_comparison['Difference'], color=colors, alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.title('Budget Variance by Category')
plt.xlabel('Category')
plt.ylabel('Difference ($)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

#### Step 6: Analyze Financial Ratios and Metrics

```python
# Calculate financial ratios
monthly_income = finance_df[finance_df['Transaction Type'] == 'Income'].groupby(pd.Grouper(key='Date', freq='M'))['Amount'].sum()
average_monthly_income = monthly_income.mean()

# Expense ratio (expenses / income)
expense_ratio = abs(total_expenses) / total_income
print(f"\nExpense Ratio: {expense_ratio:.2f} ({expense_ratio*100:.2f}%)")

# Savings ratio (savings / income)
print(f"Savings Ratio: {savings_rate/100:.2f} ({savings_rate:.2f}%)")

# Housing expense ratio (housing / income)
housing_expenses = expenses[expenses['Category'] == 'Housing']['Absolute Amount'].sum()
housing_ratio = housing_expenses / total_income
print(f"Housing Expense Ratio: {housing_ratio:.2f} ({housing_ratio*100:.2f}%)")

# Discretionary expense ratio (entertainment + shopping / income)
discretionary_categories = ['Entertainment', 'Shopping']
discretionary_expenses = expenses[expenses['Category'].isin(discretionary_categories)]['Absolute Amount'].sum()
discretionary_ratio = discretionary_expenses / total_income
print(f"Discretionary Expense Ratio: {discretionary_ratio:.2f} ({discretionary_ratio*100:.2f}%)")

# Calculate monthly cash flow
monthly_cash_flow = monthly_income + expenses.groupby(pd.Grouper(key='Date', freq='M'))['Amount'].sum()
average_cash_flow = monthly_cash_flow.mean()
print(f"Average Monthly Cash Flow: ${average_cash_flow:.2f}")

# Visualize monthly cash flow
plt.figure(figsize=(12, 6))
plt.plot(monthly_cash_flow.index, monthly_cash_flow.values, marker='o', linestyle='-', color='blue')
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
plt.title('Monthly Cash Flow')
plt.xlabel('Month')
plt.ylabel('Amount ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

#### Step 7: Create a Financial Dashboard

```python
# Create a comprehensive financial dashboard
plt.figure(figsize=(20, 15))

# Monthly income and expenses
plt.subplot(3, 2, 1)
plt.bar(monthly_summary['Month'], monthly_summary['Income'], color='green', alpha=0.7, label='Income')
plt.bar(monthly_summary['Month'], monthly_summary['Expense'], color='red', alpha=0.7, label='Expenses')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.title('Monthly Income and Expenses')
plt.xlabel('Month')
plt.ylabel('Amount ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Expenses by category
plt.subplot(3, 2, 2)
sns.barplot(x=category_expenses.index, y=category_expenses.values)
plt.title('Expenses by Category')
plt.xlabel('Category')
plt.ylabel('Amount ($)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Budget vs. actual
plt.subplot(3, 2, 3)
x = range(len(budget_comparison))
width = 0.35
plt.bar([i - width/2 for i in x], budget_comparison['Budget'], width, label='Budget', color='blue', alpha=0.7)
plt.bar([i + width/2 for i in x], budget_comparison['Actual'], width, label='Actual', color='orange', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.xticks(x, budget_comparison['Category'], rotation=45)
plt.title('Budget vs. Actual Expenses')
plt.xlabel('Category')
plt.ylabel('Amount ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Monthly cash flow
plt.subplot(3, 2, 4)
plt.plot(monthly_cash_flow.index, monthly_cash_flow.values, marker='o', linestyle='-', color='blue')
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
plt.title('Monthly Cash Flow')
plt.xlabel('Month')
plt.ylabel('Amount ($)')
plt.grid(True, alpha=0.3)

# Income sources
plt.subplot(3, 2, 5)
plt.pie(income_sources, labels=income_sources.index, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Income Sources')

# Financial ratios
plt.subplot(3, 2, 6)
ratios = {
    'Expense Ratio': expense_ratio * 100,
    'Savings Ratio': savings_rate,
    'Housing Ratio': housing_ratio * 100,
    'Discretionary Ratio': discretionary_ratio * 100
}
colors = ['red', 'green', 'blue', 'orange']
plt.bar(ratios.keys(), ratios.values(), color=colors, alpha=0.7)
plt.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% Threshold')
plt.title('Financial Ratios')
plt.xlabel('Ratio')
plt.ylabel('Percentage (%)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('financial_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Project Extensions
- Add forecasting to predict future income and expenses
- Create an interactive dashboard with filtering capabilities
- Implement budget optimization recommendations
- Add investment portfolio analysis
- Develop a financial health score based on various metrics

## Key Takeaways

- Practical projects are essential for solidifying data analysis skills
- Each project should follow a systematic approach: load data, explore, clean, analyze, visualize, and draw conclusions
- Visualization is a powerful tool for understanding patterns and communicating insights
- Real-world datasets often require significant cleaning and preprocessing
- Combining multiple analysis techniques provides more comprehensive insights
- Creating dashboards helps present findings in an organized and accessible way
- Practice projects should gradually increase in complexity as your skills develop
- Extending projects with additional analyses helps deepen your understanding
