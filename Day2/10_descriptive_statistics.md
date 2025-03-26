# Descriptive Statistics Fundamentals

## Introduction to Descriptive Statistics

Descriptive statistics summarize and describe the main features of a dataset. They provide simple summaries about the sample and the measures, forming the basis of virtually every quantitative analysis of data. This guide covers the fundamental concepts of descriptive statistics, including measures of central tendency, dispersion, distributions, percentiles, and statistical summaries.

## Measures of Central Tendency

Measures of central tendency identify the "center" of a data distribution. The three most common measures are the mean, median, and mode.

### Mean

The mean (or average) is the sum of all values divided by the number of values.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample data
data = [12, 15, 18, 22, 25, 27, 30, 32, 35, 40]

# Calculate mean
mean_value = np.mean(data)
print(f"Mean: {mean_value}")

# Using pandas
df = pd.DataFrame({'values': data})
print(f"Mean (pandas): {df['values'].mean()}")

# Visualize mean
plt.figure(figsize=(10, 6))
plt.hist(data, bins=8, alpha=0.7, color='skyblue')
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value}')
plt.title('Histogram with Mean')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

#### Properties of the Mean

- It takes into account every value in the dataset
- It's sensitive to outliers
- It's suitable for normally distributed data
- It can be used for further statistical calculations

### Median

The median is the middle value when the data is arranged in order. If there's an even number of observations, the median is the average of the two middle values.

```python
# Calculate median
median_value = np.median(data)
print(f"Median: {median_value}")

# Using pandas
print(f"Median (pandas): {df['values'].median()}")

# Visualize median
plt.figure(figsize=(10, 6))
plt.hist(data, bins=8, alpha=0.7, color='skyblue')
plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value}')
plt.title('Histogram with Median')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

#### Properties of the Median

- It's not affected by outliers
- It's suitable for skewed distributions
- It's useful when the data contains extreme values
- It represents the 50th percentile of the data

### Mode

The mode is the most frequently occurring value in the dataset.

```python
# Create data with a clear mode
data_with_mode = [12, 15, 18, 22, 22, 22, 25, 27, 30, 32, 35, 40]

# Calculate mode
from scipy import stats
mode_value = stats.mode(data_with_mode)[0][0]
print(f"Mode: {mode_value}")

# Using pandas
print(f"Mode (pandas): {pd.Series(data_with_mode).mode()[0]}")

# Visualize mode
plt.figure(figsize=(10, 6))
plt.hist(data_with_mode, bins=8, alpha=0.7, color='skyblue')
plt.axvline(mode_value, color='purple', linestyle='dashed', linewidth=2, label=f'Mode: {mode_value}')
plt.title('Histogram with Mode')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

#### Properties of the Mode

- It can be used for both numerical and categorical data
- A dataset can have multiple modes (bimodal, multimodal)
- It's not affected by outliers
- It represents the most common value in the dataset

### Comparing Measures of Central Tendency

```python
# Create a skewed dataset
skewed_data = [10, 12, 14, 15, 16, 18, 20, 25, 40, 80]

# Calculate measures
mean_skewed = np.mean(skewed_data)
median_skewed = np.median(skewed_data)
mode_skewed = stats.mode(skewed_data)[0][0]

# Visualize comparison
plt.figure(figsize=(10, 6))
plt.hist(skewed_data, bins=8, alpha=0.7, color='skyblue')
plt.axvline(mean_skewed, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_skewed}')
plt.axvline(median_skewed, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_skewed}')
plt.axvline(mode_skewed, color='purple', linestyle='dashed', linewidth=2, label=f'Mode: {mode_skewed}')
plt.title('Comparison of Mean, Median, and Mode in Skewed Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

## Measures of Dispersion

Measures of dispersion describe the spread or variability of the data.

### Variance

Variance measures how far each value in the dataset is from the mean.

```python
# Calculate variance
variance = np.var(data)
print(f"Variance: {variance}")

# Using pandas
print(f"Variance (pandas): {df['values'].var()}")

# Note: NumPy's var() calculates the population variance by default
# For sample variance, use ddof=1
sample_variance = np.var(data, ddof=1)
print(f"Sample Variance: {sample_variance}")
```

#### Properties of Variance

- It's always non-negative
- A larger variance indicates more spread in the data
- It's sensitive to outliers
- It's expressed in squared units of the original data

### Standard Deviation

Standard deviation is the square root of the variance, providing a measure of dispersion in the same units as the original data.

```python
# Calculate standard deviation
std_dev = np.std(data)
print(f"Standard Deviation: {std_dev}")

# Using pandas
print(f"Standard Deviation (pandas): {df['values'].std()}")

# Sample standard deviation
sample_std = np.std(data, ddof=1)
print(f"Sample Standard Deviation: {sample_std}")

# Visualize standard deviation
plt.figure(figsize=(10, 6))
plt.hist(data, bins=8, alpha=0.7, color='skyblue')
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value}')
plt.axvline(mean_value + std_dev, color='orange', linestyle='dashed', linewidth=2, label=f'Mean + SD: {mean_value + std_dev}')
plt.axvline(mean_value - std_dev, color='orange', linestyle='dashed', linewidth=2, label=f'Mean - SD: {mean_value - std_dev}')
plt.title('Histogram with Standard Deviation')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

#### Properties of Standard Deviation

- It's in the same units as the original data
- For normally distributed data, approximately 68% of values fall within one standard deviation of the mean
- It's commonly used to measure confidence in statistical conclusions
- Like variance, it's sensitive to outliers

### Range

The range is the difference between the maximum and minimum values in the dataset.

```python
# Calculate range
data_range = np.max(data) - np.min(data)
print(f"Range: {data_range}")

# Using pandas
print(f"Range (pandas): {df['values'].max() - df['values'].min()}")
```

#### Properties of Range

- It's simple to calculate and understand
- It only considers the two extreme values
- It's highly sensitive to outliers
- It doesn't provide information about the distribution between the extremes

### Interquartile Range (IQR)

The interquartile range is the difference between the 75th and 25th percentiles of the data.

```python
# Calculate IQR
q75, q25 = np.percentile(data, [75, 25])
iqr = q75 - q25
print(f"Interquartile Range: {iqr}")

# Using pandas
print(f"IQR (pandas): {df['values'].quantile(0.75) - df['values'].quantile(0.25)}")

# Visualize IQR with box plot
plt.figure(figsize=(10, 6))
plt.boxplot(data, vert=False)
plt.title('Box Plot Showing IQR')
plt.xlabel('Value')
plt.tight_layout()
plt.show()
```

#### Properties of IQR

- It's not affected by outliers
- It describes the middle 50% of the data
- It's used to identify outliers (values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR)
- It's a robust measure of dispersion

## Understanding Distributions

The distribution of data describes the overall pattern of values.

### Types of Distributions

```python
# Create different distributions
np.random.seed(42)
normal_dist = np.random.normal(50, 10, 1000)  # Normal distribution
skewed_right = np.random.exponential(10, 1000)  # Right-skewed
skewed_left = 100 - np.random.exponential(10, 1000)  # Left-skewed
uniform_dist = np.random.uniform(0, 100, 1000)  # Uniform distribution
bimodal_dist = np.concatenate([np.random.normal(30, 5, 500), np.random.normal(70, 5, 500)])  # Bimodal

# Visualize distributions
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.flatten()

# Normal distribution
sns.histplot(normal_dist, kde=True, ax=axes[0])
axes[0].set_title('Normal Distribution')
axes[0].axvline(np.mean(normal_dist), color='red', linestyle='dashed', label='Mean')
axes[0].axvline(np.median(normal_dist), color='green', linestyle='dashed', label='Median')
axes[0].legend()

# Right-skewed distribution
sns.histplot(skewed_right, kde=True, ax=axes[1])
axes[1].set_title('Right-Skewed Distribution')
axes[1].axvline(np.mean(skewed_right), color='red', linestyle='dashed', label='Mean')
axes[1].axvline(np.median(skewed_right), color='green', linestyle='dashed', label='Median')
axes[1].legend()

# Left-skewed distribution
sns.histplot(skewed_left, kde=True, ax=axes[2])
axes[2].set_title('Left-Skewed Distribution')
axes[2].axvline(np.mean(skewed_left), color='red', linestyle='dashed', label='Mean')
axes[2].axvline(np.median(skewed_left), color='green', linestyle='dashed', label='Median')
axes[2].legend()

# Uniform distribution
sns.histplot(uniform_dist, kde=True, ax=axes[3])
axes[3].set_title('Uniform Distribution')
axes[3].axvline(np.mean(uniform_dist), color='red', linestyle='dashed', label='Mean')
axes[3].axvline(np.median(uniform_dist), color='green', linestyle='dashed', label='Median')
axes[3].legend()

# Bimodal distribution
sns.histplot(bimodal_dist, kde=True, ax=axes[4])
axes[4].set_title('Bimodal Distribution')
axes[4].axvline(np.mean(bimodal_dist), color='red', linestyle='dashed', label='Mean')
axes[4].axvline(np.median(bimodal_dist), color='green', linestyle='dashed', label='Median')
axes[4].legend()

# Remove empty subplot
axes[5].set_visible(False)

plt.tight_layout()
plt.show()
```

### Measuring Skewness and Kurtosis

Skewness measures the asymmetry of the distribution, while kurtosis measures the "tailedness" of the distribution.

```python
from scipy.stats import skew, kurtosis

# Calculate skewness
normal_skew = skew(normal_dist)
right_skew = skew(skewed_right)
left_skew = skew(skewed_left)

print(f"Skewness of normal distribution: {normal_skew}")
print(f"Skewness of right-skewed distribution: {right_skew}")
print(f"Skewness of left-skewed distribution: {left_skew}")

# Calculate kurtosis
normal_kurt = kurtosis(normal_dist)
right_kurt = kurtosis(skewed_right)
bimodal_kurt = kurtosis(bimodal_dist)

print(f"Kurtosis of normal distribution: {normal_kurt}")
print(f"Kurtosis of right-skewed distribution: {right_kurt}")
print(f"Kurtosis of bimodal distribution: {bimodal_kurt}")
```

#### Interpreting Skewness

- **Skewness = 0**: Symmetric distribution
- **Skewness > 0**: Right-skewed (positive skew)
- **Skewness < 0**: Left-skewed (negative skew)

#### Interpreting Kurtosis

- **Kurtosis = 0**: Normal distribution (mesokurtic)
- **Kurtosis > 0**: Heavy-tailed distribution (leptokurtic)
- **Kurtosis < 0**: Light-tailed distribution (platykurtic)

## Percentiles and Quartiles

Percentiles divide the data into 100 equal parts, while quartiles divide it into 4 equal parts.

### Calculating Percentiles

```python
# Calculate percentiles
p25 = np.percentile(data, 25)
p50 = np.percentile(data, 50)  # Same as median
p75 = np.percentile(data, 75)
p90 = np.percentile(data, 90)
p95 = np.percentile(data, 95)

print(f"25th percentile: {p25}")
print(f"50th percentile (median): {p50}")
print(f"75th percentile: {p75}")
print(f"90th percentile: {p90}")
print(f"95th percentile: {p95}")

# Using pandas
print(f"Percentiles (pandas):")
print(df['values'].quantile([0.25, 0.5, 0.75, 0.9, 0.95]))
```

### Visualizing Quartiles

```python
# Visualize quartiles
plt.figure(figsize=(10, 6))
plt.hist(data, bins=8, alpha=0.7, color='skyblue')
plt.axvline(p25, color='green', linestyle='dashed', linewidth=2, label=f'Q1 (25%): {p25}')
plt.axvline(p50, color='red', linestyle='dashed', linewidth=2, label=f'Q2 (50%): {p50}')
plt.axvline(p75, color='purple', linestyle='dashed', linewidth=2, label=f'Q3 (75%): {p75}')
plt.title('Histogram with Quartiles')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

### Box Plots for Visualizing Quartiles

```python
# Create box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x=data)
plt.title('Box Plot Showing Quartiles')
plt.xlabel('Value')
plt.tight_layout()
plt.show()
```

## Statistical Summaries with Pandas

Pandas provides convenient methods for generating statistical summaries of data.

### Using describe()

```python
# Create a DataFrame with multiple variables
np.random.seed(42)
df_multi = pd.DataFrame({
    'A': np.random.normal(50, 10, 100),
    'B': np.random.exponential(10, 100),
    'C': np.random.uniform(0, 100, 100),
    'D': np.random.choice(['X', 'Y', 'Z'], 100)
})

# Generate descriptive statistics
desc_stats = df_multi.describe()
print(desc_stats)

# Include all columns (including categorical)
desc_stats_all = df_multi.describe(include='all')
print(desc_stats_all)
```

### Custom Statistical Summaries

```python
# Custom summary statistics
custom_stats = df_multi.agg({
    'A': ['min', 'max', 'mean', 'median', 'std', 'var', 'skew', 'kurt'],
    'B': ['min', 'max', 'mean', 'median', 'std', 'var', 'skew', 'kurt'],
    'C': ['min', 'max', 'mean', 'median', 'std', 'var', 'skew', 'kurt'],
    'D': ['count', 'nunique', lambda x: x.value_counts().to_dict()]
})
print(custom_stats)
```

### Visualizing Summary Statistics

```python
# Visualize summary statistics
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Histogram for A (normal)
sns.histplot(df_multi['A'], kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of A (Normal)')
axes[0, 0].axvline(df_multi['A'].mean(), color='red', linestyle='dashed', label='Mean')
axes[0, 0].axvline(df_multi['A'].median(), color='green', linestyle='dashed', label='Median')
axes[0, 0].legend()

# Histogram for B (exponential)
sns.histplot(df_multi['B'], kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Distribution of B (Exponential)')
axes[0, 1].axvline(df_multi['B'].mean(), color='red', linestyle='dashed', label='Mean')
axes[0, 1].axvline(df_multi['B'].median(), color='green', linestyle='dashed', label='Median')
axes[0, 1].legend()

# Histogram for C (uniform)
sns.histplot(df_multi['C'], kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Distribution of C (Uniform)')
axes[1, 0].axvline(df_multi['C'].mean(), color='red', linestyle='dashed', label='Mean')
axes[1, 0].axvline(df_multi['C'].median(), color='green', linestyle='dashed', label='Median')
axes[1, 0].legend()

# Bar chart for D (categorical)
sns.countplot(x='D', data=df_multi, ax=axes[1, 1])
axes[1, 1].set_title('Count of D (Categorical)')

plt.tight_layout()
plt.show()
```

## Comprehensive Example: Analyzing a Real Dataset

Let's apply descriptive statistics to a real dataset.

```python
# Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display the first few rows
print(iris_df.head())

# Generate descriptive statistics
print(iris_df.describe())

# Descriptive statistics by group
grouped_stats = iris_df.groupby('species').describe()
print(grouped_stats)

# Visualize distributions
plt.figure(figsize=(15, 10))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i+1)
    for species in iris.target_names:
        subset = iris_df[iris_df['species'] == species]
        sns.histplot(subset[feature], kde=True, label=species)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.legend()
plt.tight_layout()
plt.show()

# Box plots by species
plt.figure(figsize=(15, 10))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='species', y=feature, data=iris_df)
    plt.title(f'Box Plot of {feature} by Species')
plt.tight_layout()
plt.show()
```

## Practice Exercises

1. Calculate and interpret the mean, median, and mode for a dataset of your choice.
2. Compare the standard deviation and IQR for a normally distributed dataset and a skewed dataset.
3. Create histograms for different types of distributions and identify their characteristics.
4. Calculate percentiles for a dataset and interpret what they tell you about the data.
5. Generate a comprehensive statistical summary for a multi-variable dataset.
6. Create box plots to visualize the distribution of data across different categories.
7. Analyze the skewness and kurtosis of different distributions and explain what they indicate.
8. Compare the central tendency and dispersion measures for different groups within a dataset.

## Key Takeaways

- Measures of central tendency (mean, median, mode) identify the "center" of a data distribution
- Measures of dispersion (variance, standard deviation, range, IQR) describe the spread of the data
- Different types of distributions (normal, skewed, uniform, bimodal) have different characteristics
- Percentiles and quartiles divide the data into equal parts and help understand the distribution
- Statistical summaries provide a comprehensive overview of the data's main features
- Visualization tools like histograms and box plots help interpret descriptive statistics
- The choice of descriptive statistics depends on the data type and distribution
- Descriptive statistics form the foundation for more advanced statistical analyses
