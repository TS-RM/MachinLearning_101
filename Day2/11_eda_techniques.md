# EDA Techniques and Workflows

## Introduction to Exploratory Data Analysis

Exploratory Data Analysis (EDA) is an approach to analyzing datasets to summarize their main characteristics, often using visual methods. It's a critical step in the data analysis process that helps you understand the structure, patterns, relationships, and anomalies in your data before applying more complex techniques. This guide covers systematic EDA approaches, data profiling tools, and techniques for univariate, bivariate, and multivariate analysis.

## Developing a Systematic EDA Approach

A systematic approach to EDA ensures that you thoroughly explore your data and don't miss important insights.

### The EDA Process

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for visualizations
sns.set_style('whitegrid')
```

#### Step 1: Understand the Data Context

Before diving into the data, understand:
- What does each variable represent?
- What are the units of measurement?
- How was the data collected?
- What is the time period covered?
- What are the expected relationships?

#### Step 2: Data Overview

```python
# Load a sample dataset
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# Get a quick overview
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Data types and non-null counts
print("\nData information:")
print(df.info())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())
```

#### Step 3: Data Cleaning Check

```python
# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Check for duplicates
print("\nNumber of duplicate rows:", df.duplicated().sum())

# Check for outliers using box plots
plt.figure(figsize=(15, 10))
df.boxplot()
plt.title('Box Plots to Check for Outliers')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
```

#### Step 4: Univariate Analysis

Examine each variable individually:

```python
# Histograms for all numeric variables
plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns):
    plt.subplot(3, 4, i+1)
    df[column].hist(bins=20)
    plt.title(column)
plt.tight_layout()
plt.show()

# Descriptive statistics for each variable
for column in df.columns:
    print(f"\nStatistics for {column}:")
    print(f"Mean: {df[column].mean()}")
    print(f"Median: {df[column].median()}")
    print(f"Std Dev: {df[column].std()}")
    print(f"Min: {df[column].min()}")
    print(f"Max: {df[column].max()}")
    print(f"Skewness: {df[column].skew()}")
    print(f"Kurtosis: {df[column].kurtosis()}")
```

#### Step 5: Bivariate Analysis

Examine relationships between pairs of variables:

```python
# Correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Scatter plots for selected pairs
plt.figure(figsize=(15, 10))
for i, x_var in enumerate(['bmi', 'bp', 's1']):
    for j, y_var in enumerate(['target']):
        plt.subplot(1, 3, i+1)
        plt.scatter(df[x_var], df[y_var], alpha=0.5)
        plt.title(f'{x_var} vs {y_var}')
        plt.xlabel(x_var)
        plt.ylabel(y_var)
plt.tight_layout()
plt.show()
```

#### Step 6: Multivariate Analysis

Examine relationships between multiple variables:

```python
# Pair plot for selected variables
selected_vars = ['bmi', 'bp', 's1', 's2', 'target']
sns.pairplot(df[selected_vars], diag_kind='kde')
plt.suptitle('Pair Plot of Selected Variables', y=1.02)
plt.show()

# 3D scatter plot for three variables
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['bmi'], df['bp'], df['target'], c=df['target'], cmap='viridis', s=50, alpha=0.6)
ax.set_xlabel('BMI')
ax.set_ylabel('Blood Pressure')
ax.set_zlabel('Target')
plt.title('3D Scatter Plot')
plt.tight_layout()
plt.show()
```

#### Step 7: Hypothesis Generation

Based on your EDA, formulate hypotheses about the data:
- Which variables seem most related to the target?
- Are there any unexpected patterns or relationships?
- What further analyses might be valuable?

### EDA Checklist

Create a checklist to ensure a thorough exploration:

1. **Data Understanding**
   - [ ] Understand the meaning of each variable
   - [ ] Identify the data types
   - [ ] Understand the context and collection method

2. **Data Quality Check**
   - [ ] Check for missing values
   - [ ] Identify duplicates
   - [ ] Detect outliers
   - [ ] Verify data consistency

3. **Univariate Analysis**
   - [ ] Examine distributions of all variables
   - [ ] Calculate descriptive statistics
   - [ ] Identify skewness and kurtosis

4. **Bivariate Analysis**
   - [ ] Calculate correlations between variables
   - [ ] Create scatter plots for important pairs
   - [ ] Analyze categorical relationships

5. **Multivariate Analysis**
   - [ ] Examine relationships between multiple variables
   - [ ] Look for interaction effects
   - [ ] Identify complex patterns

6. **Hypothesis Generation**
   - [ ] Formulate hypotheses based on findings
   - [ ] Identify areas for deeper analysis
   - [ ] Plan next steps for modeling or further investigation

## Data Profiling Tools

Data profiling tools automate much of the EDA process, providing comprehensive reports about your data.

### Using pandas-profiling (ydata-profiling)

```python
# Install the package if not already installed
# !pip install ydata-profiling

from ydata_profiling import ProfileReport

# Generate a profile report
profile = ProfileReport(df, title="Diabetes Dataset Profiling Report", explorative=True)

# Display the report (in a Jupyter notebook)
# profile.to_notebook_iframe()

# Save the report to a file
profile.to_file("diabetes_profile_report.html")
```

The profile report includes:
- Overview of the dataset
- Variables information
- Interactions between variables
- Correlations
- Missing values
- Distributions
- Descriptive statistics

### Using sweetviz

```python
# Install the package if not already installed
# !pip install sweetviz

import sweetviz as sv

# Generate a report
report = sv.analyze(df)

# Display the report (in a Jupyter notebook)
# report.show_notebook()

# Save the report to a file
report.show_html("diabetes_sweetviz_report.html")
```

### Using D-Tale

```python
# Install the package if not already installed
# !pip install dtale

import dtale

# Launch D-Tale with your dataframe
d = dtale.show(df)

# Get the URL to access D-Tale
# print(d._url)
```

### Custom Profiling Functions

You can also create your own profiling functions for specific needs:

```python
def custom_data_profile(df):
    """Generate a custom data profile report."""
    report = {
        'shape': df.shape,
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'numeric_summary': df.describe().to_dict(),
        'categorical_summary': {col: df[col].value_counts().to_dict() 
                               for col in df.select_dtypes(include=['object', 'category']).columns},
        'correlations': df.corr()['target'].sort_values(ascending=False).to_dict() if 'target' in df.columns else None
    }
    
    return report

# Generate and print the custom profile
custom_profile = custom_data_profile(df)
for section, content in custom_profile.items():
    print(f"\n{section.upper()}:")
    print(content)
```

## Univariate Analysis Techniques

Univariate analysis examines variables one at a time.

### Analyzing Numeric Variables

```python
# Select a numeric variable
variable = 'bmi'

# Descriptive statistics
print(f"Statistics for {variable}:")
print(df[variable].describe())

# Histogram
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df[variable], bins=20, color='skyblue', edgecolor='black')
plt.title(f'Histogram of {variable}')
plt.xlabel(variable)
plt.ylabel('Frequency')

# Kernel Density Estimate (KDE) plot
plt.subplot(1, 2, 2)
sns.kdeplot(df[variable], fill=True)
plt.axvline(df[variable].mean(), color='red', linestyle='--', label='Mean')
plt.axvline(df[variable].median(), color='green', linestyle='--', label='Median')
plt.title(f'Density Plot of {variable}')
plt.xlabel(variable)
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()

# Box plot
plt.figure(figsize=(8, 6))
sns.boxplot(y=df[variable])
plt.title(f'Box Plot of {variable}')
plt.ylabel(variable)
plt.show()

# QQ plot to check for normality
from scipy import stats
import statsmodels.api as sm

plt.figure(figsize=(8, 6))
sm.qqplot(df[variable], line='45')
plt.title(f'QQ Plot of {variable}')
plt.show()

# Test for normality
stat, p_value = stats.shapiro(df[variable])
print(f"Shapiro-Wilk Test for Normality:")
print(f"Statistic: {stat}, p-value: {p_value}")
print(f"The data is {'normally distributed' if p_value > 0.05 else 'not normally distributed'}")
```

### Analyzing Categorical Variables

```python
# For this example, let's create a categorical variable
df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 25, 30, 100], labels=['Normal', 'Overweight', 'Obese'])

# Frequency table
freq_table = df['bmi_category'].value_counts()
print("Frequency Table:")
print(freq_table)

# Percentage table
percent_table = df['bmi_category'].value_counts(normalize=True) * 100
print("\nPercentage Table:")
print(percent_table)

# Bar chart
plt.figure(figsize=(10, 6))
sns.countplot(x='bmi_category', data=df)
plt.title('Count of BMI Categories')
plt.xlabel('BMI Category')
plt.ylabel('Count')
plt.show()

# Pie chart
plt.figure(figsize=(10, 6))
plt.pie(freq_table, labels=freq_table.index, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Distribution of BMI Categories')
plt.show()
```

## Bivariate Analysis for Relationships

Bivariate analysis examines the relationship between two variables.

### Numeric vs. Numeric

```python
# Select two numeric variables
x_var = 'bmi'
y_var = 'target'

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df[x_var], df[y_var], alpha=0.5)
plt.title(f'{x_var} vs {y_var}')
plt.xlabel(x_var)
plt.ylabel(y_var)
plt.grid(True, alpha=0.3)
plt.show()

# Add regression line
plt.figure(figsize=(10, 6))
sns.regplot(x=x_var, y=y_var, data=df, scatter_kws={'alpha':0.5})
plt.title(f'{x_var} vs {y_var} with Regression Line')
plt.xlabel(x_var)
plt.ylabel(y_var)
plt.grid(True, alpha=0.3)
plt.show()

# Calculate correlation
correlation = df[[x_var, y_var]].corr().iloc[0, 1]
print(f"Correlation between {x_var} and {y_var}: {correlation:.4f}")

# Test for significance of correlation
from scipy.stats import pearsonr
corr, p_value = pearsonr(df[x_var], df[y_var])
print(f"Pearson correlation: {corr:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"The correlation is {'statistically significant' if p_value < 0.05 else 'not statistically significant'}")

# Hexbin plot for large datasets
plt.figure(figsize=(10, 6))
plt.hexbin(df[x_var], df[y_var], gridsize=20, cmap='Blues')
plt.colorbar(label='Count')
plt.title(f'Hexbin Plot of {x_var} vs {y_var}')
plt.xlabel(x_var)
plt.ylabel(y_var)
plt.show()
```

### Numeric vs. Categorical

```python
# Select a numeric and a categorical variable
num_var = 'target'
cat_var = 'bmi_category'

# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x=cat_var, y=num_var, data=df)
plt.title(f'{num_var} by {cat_var}')
plt.xlabel(cat_var)
plt.ylabel(num_var)
plt.show()

# Violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x=cat_var, y=num_var, data=df, inner='quartile')
plt.title(f'{num_var} by {cat_var} (Violin Plot)')
plt.xlabel(cat_var)
plt.ylabel(num_var)
plt.show()

# Bar plot of means with error bars
plt.figure(figsize=(10, 6))
sns.barplot(x=cat_var, y=num_var, data=df, ci=95)
plt.title(f'Mean {num_var} by {cat_var} with 95% CI')
plt.xlabel(cat_var)
plt.ylabel(f'Mean {num_var}')
plt.show()

# ANOVA test to compare means across categories
from scipy.stats import f_oneway

# Group data by category
groups = [df[df[cat_var] == cat][num_var] for cat in df[cat_var].unique()]

# Perform ANOVA
f_stat, p_value = f_oneway(*groups)
print(f"ANOVA Results:")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"There {'is' if p_value < 0.05 else 'is not'} a statistically significant difference in means across categories")
```

### Categorical vs. Categorical

```python
# Create another categorical variable for demonstration
df['bp_category'] = pd.cut(df['bp'], bins=[0, 80, 120, 200], labels=['Low', 'Normal', 'High'])

# Select two categorical variables
cat_var1 = 'bmi_category'
cat_var2 = 'bp_category'

# Cross-tabulation (contingency table)
contingency_table = pd.crosstab(df[cat_var1], df[cat_var2])
print("Contingency Table (Counts):")
print(contingency_table)

# Cross-tabulation with percentages
contingency_pct = pd.crosstab(df[cat_var1], df[cat_var2], normalize='index') * 100
print("\nContingency Table (Row Percentages):")
print(contingency_pct)

# Stacked bar chart
plt.figure(figsize=(10, 6))
contingency_table.plot(kind='bar', stacked=True)
plt.title(f'{cat_var1} vs {cat_var2}')
plt.xlabel(cat_var1)
plt.ylabel('Count')
plt.legend(title=cat_var2)
plt.show()

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, cmap='YlGnBu', fmt='d')
plt.title(f'Heatmap of {cat_var1} vs {cat_var2}')
plt.show()

# Chi-square test for independence
from scipy.stats import chi2_contingency
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square Test Results:")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"The variables are {'dependent' if p_value < 0.05 else 'independent'}")
```

## Multivariate Analysis for Complex Patterns

Multivariate analysis examines relationships between three or more variables.

### Scatter Plot Matrix

```python
# Select multiple variables
selected_vars = ['bmi', 'bp', 's1', 's2', 'target']

# Scatter plot matrix
sns.pairplot(df[selected_vars], diag_kind='kde')
plt.suptitle('Scatter Plot Matrix', y=1.02)
plt.show()
```

### Correlation Heatmap

```python
# Correlation matrix
correlation_matrix = df[selected_vars].corr()

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
```

### Parallel Coordinates Plot

```python
# Normalize the data for parallel coordinates
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[selected_vars]), columns=selected_vars)

# Add a categorical variable for coloring
df_scaled['target_category'] = pd.cut(df['target'], bins=3, labels=['Low', 'Medium', 'High'])

# Parallel coordinates plot
plt.figure(figsize=(12, 6))
pd.plotting.parallel_coordinates(df_scaled, 'target_category', colormap='viridis')
plt.title('Parallel Coordinates Plot')
plt.grid(False)
plt.show()
```

### 3D Scatter Plot

```python
# 3D scatter plot with color for a fourth variable
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x = df['bmi']
y = df['bp']
z = df['s1']
colors = df['target']

scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=50, alpha=0.6)
ax.set_xlabel('BMI')
ax.set_ylabel('Blood Pressure')
ax.set_zlabel('S1')
plt.colorbar(scatter, label='Target')
plt.title('3D Scatter Plot')
plt.tight_layout()
plt.show()
```

### Clustering for Pattern Detection

```python
# Select variables for clustering
cluster_vars = ['bmi', 'bp', 's1', 's2']
X = df[cluster_vars]

# Standardize the data
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

# Apply K-means clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Visualize clusters in 2D using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis', s=50, alpha=0.8)
plt.colorbar(scatter, label='Cluster')
plt.title('PCA Projection of Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Analyze clusters
cluster_analysis = df.groupby('cluster')[cluster_vars + ['target']].mean()
print("Cluster Analysis:")
print(cluster_analysis)
```

## Comprehensive EDA Example

Let's put everything together with a comprehensive EDA on a real dataset.

```python
# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Step 1: Data Overview
print("Dataset shape:", titanic.shape)
print("\nFirst few rows:")
print(titanic.head())
print("\nData information:")
print(titanic.info())
print("\nSummary statistics:")
print(titanic.describe())

# Step 2: Data Cleaning Check
print("\nMissing values per column:")
print(titanic.isnull().sum())
print("\nNumber of duplicate rows:", titanic.duplicated().sum())

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(titanic.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.tight_layout()
plt.show()

# Step 3: Univariate Analysis

# Numeric variables
numeric_vars = titanic.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(15, 10))
for i, var in enumerate(numeric_vars):
    plt.subplot(2, 3, i+1)
    sns.histplot(titanic[var], kde=True)
    plt.title(f'Distribution of {var}')
plt.tight_layout()
plt.show()

# Categorical variables
categorical_vars = titanic.select_dtypes(include=['object', 'category']).columns
plt.figure(figsize=(15, 10))
for i, var in enumerate(categorical_vars):
    plt.subplot(2, 3, i+1)
    sns.countplot(y=var, data=titanic)
    plt.title(f'Count of {var}')
plt.tight_layout()
plt.show()

# Step 4: Bivariate Analysis with Survival

# Numeric vs. Survival
plt.figure(figsize=(15, 10))
for i, var in enumerate(numeric_vars):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x='survived', y=var, data=titanic)
    plt.title(f'{var} by Survival')
plt.tight_layout()
plt.show()

# Categorical vs. Survival
plt.figure(figsize=(15, 15))
for i, var in enumerate(categorical_vars):
    plt.subplot(3, 2, i+1)
    sns.countplot(x=var, hue='survived', data=titanic)
    plt.title(f'Survival by {var}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 5: Multivariate Analysis

# Age, Fare, and Survival
plt.figure(figsize=(10, 8))
sns.scatterplot(x='age', y='fare', hue='survived', size='fare', sizes=(20, 200), data=titanic)
plt.title('Age vs. Fare by Survival')
plt.show()

# Age, Class, Sex, and Survival
plt.figure(figsize=(15, 10))
sns.catplot(x='pclass', y='survived', col='sex', kind='bar', data=titanic)
plt.suptitle('Survival by Class and Sex', y=1.05)
plt.tight_layout()
plt.show()

# Step 6: Correlation Analysis
correlation_matrix = titanic.select_dtypes(include=['int64', 'float64']).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Step 7: Feature Engineering and Further Analysis
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1
titanic['is_alone'] = (titanic['family_size'] == 1).astype(int)

plt.figure(figsize=(10, 6))
sns.barplot(x='family_size', y='survived', data=titanic)
plt.title('Survival Rate by Family Size')
plt.show()

# Step 8: Hypothesis Testing
from scipy.stats import ttest_ind

# Test if fare differs significantly between survivors and non-survivors
survivors = titanic[titanic['survived'] == 1]['fare']
non_survivors = titanic[titanic['survived'] == 0]['fare']
t_stat, p_value = ttest_ind(survivors, non_survivors, equal_var=False)
print(f"T-test for fare difference between survivors and non-survivors:")
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
print(f"The difference is {'statistically significant' if p_value < 0.05 else 'not statistically significant'}")

# Step 9: Summary of Findings
print("\nKey Findings from EDA:")
print("1. There are missing values in age, cabin, and embarked columns")
print("2. Females had a higher survival rate than males")
print("3. Passengers in higher classes (1st class) had better survival rates")
print("4. Fare was higher for survivors, suggesting wealth played a role in survival")
print("5. Middle-sized families had better survival rates than singles or very large families")
```

## Practice Exercises

1. Develop a systematic EDA approach for a dataset of your choice, following the steps outlined in this guide.
2. Use a data profiling tool to generate a comprehensive report for a dataset and identify key insights.
3. Perform univariate analysis on numeric and categorical variables, interpreting the distributions and statistics.
4. Conduct bivariate analysis to explore relationships between different types of variables.
5. Apply multivariate analysis techniques to uncover complex patterns in your data.
6. Generate and test hypotheses based on your EDA findings.
7. Create a custom data profiling function tailored to your specific analysis needs.
8. Perform a complete EDA on a real-world dataset and summarize your key findings.

## Key Takeaways

- A systematic EDA approach ensures thorough exploration of your data
- Data profiling tools automate much of the EDA process, providing comprehensive reports
- Univariate analysis examines the distribution and statistics of individual variables
- Bivariate analysis explores relationships between pairs of variables
- Multivariate analysis uncovers complex patterns involving multiple variables
- Visualization is a key component of effective EDA
- Statistical tests help validate observations and hypotheses
- EDA is an iterative process that often leads to new questions and insights
- Effective EDA forms the foundation for successful modeling and analysis
