# Statistical Analysis on Real Datasets

## Introduction to Statistical Analysis

Statistical analysis allows us to make inferences about data, test hypotheses, and quantify uncertainty. When applied to real datasets, these techniques help us uncover meaningful insights and make data-driven decisions. This guide covers various statistical analysis methods including normality tests, hypothesis testing, confidence intervals, ANOVA, and non-parametric tests.

## Testing for Normality

Many statistical tests assume that data follows a normal distribution. Before applying these tests, it's important to check if this assumption is valid.

### Visual Methods for Checking Normality

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Set the style for visualizations
sns.set_style('whitegrid')

# Load a sample dataset
from sklearn.datasets import load_boston
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

# Select a variable to test for normality
variable = 'PRICE'

# Histogram with normal curve overlay
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df[variable], kde=True, stat="density")
# Add normal curve
x = np.linspace(df[variable].min(), df[variable].max(), 100)
y = stats.norm.pdf(x, df[variable].mean(), df[variable].std())
plt.plot(x, y, 'r--')
plt.title(f'Histogram of {variable} with Normal Curve')

# Q-Q plot
plt.subplot(1, 2, 2)
stats.probplot(df[variable], plot=plt)
plt.title(f'Q-Q Plot of {variable}')

plt.tight_layout()
plt.show()
```

### Statistical Tests for Normality

```python
# Shapiro-Wilk test (best for small samples, n < 50)
shapiro_stat, shapiro_p = stats.shapiro(df[variable])
print(f"Shapiro-Wilk Test for {variable}:")
print(f"Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")
print(f"Conclusion: The data {'is' if shapiro_p > 0.05 else 'is not'} normally distributed (α = 0.05)\n")

# D'Agostino-Pearson test (better for larger samples)
k2_stat, k2_p = stats.normaltest(df[variable])
print(f"D'Agostino-Pearson Test for {variable}:")
print(f"Statistic: {k2_stat:.4f}, p-value: {k2_p:.4f}")
print(f"Conclusion: The data {'is' if k2_p > 0.05 else 'is not'} normally distributed (α = 0.05)\n")

# Kolmogorov-Smirnov test
ks_stat, ks_p = stats.kstest(df[variable], 'norm', args=(df[variable].mean(), df[variable].std()))
print(f"Kolmogorov-Smirnov Test for {variable}:")
print(f"Statistic: {ks_stat:.4f}, p-value: {ks_p:.4f}")
print(f"Conclusion: The data {'is' if ks_p > 0.05 else 'is not'} normally distributed (α = 0.05)")
```

### Transforming Non-Normal Data

If your data isn't normally distributed, you can apply transformations to make it more normal:

```python
# Common transformations for right-skewed data
plt.figure(figsize=(15, 10))

# Original data
plt.subplot(2, 3, 1)
sns.histplot(df[variable], kde=True)
plt.title(f'Original {variable}')

# Log transformation
plt.subplot(2, 3, 2)
log_var = np.log(df[variable])
sns.histplot(log_var, kde=True)
plt.title(f'Log of {variable}')

# Square root transformation
plt.subplot(2, 3, 3)
sqrt_var = np.sqrt(df[variable])
sns.histplot(sqrt_var, kde=True)
plt.title(f'Square Root of {variable}')

# Box-Cox transformation
from scipy.stats import boxcox
plt.subplot(2, 3, 4)
boxcox_var, lambda_value = boxcox(df[variable])
sns.histplot(boxcox_var, kde=True)
plt.title(f'Box-Cox of {variable} (λ={lambda_value:.2f})')

# Reciprocal transformation
plt.subplot(2, 3, 5)
recip_var = 1 / df[variable]
sns.histplot(recip_var, kde=True)
plt.title(f'Reciprocal of {variable}')

# Yeo-Johnson transformation (works with negative values too)
from sklearn.preprocessing import PowerTransformer
plt.subplot(2, 3, 6)
pt = PowerTransformer(method='yeo-johnson')
yj_var = pt.fit_transform(df[[variable]])
sns.histplot(yj_var, kde=True)
plt.title(f'Yeo-Johnson of {variable}')

plt.tight_layout()
plt.show()

# Test normality of transformed data
print("Shapiro-Wilk Test for Transformed Variables:")
for name, data in [('Original', df[variable]), 
                   ('Log', log_var), 
                   ('Square Root', sqrt_var), 
                   ('Box-Cox', boxcox_var),
                   ('Reciprocal', recip_var),
                   ('Yeo-Johnson', yj_var.flatten())]:
    stat, p = stats.shapiro(data)
    print(f"{name}: Statistic = {stat:.4f}, p-value = {p:.4f}, {'Normal' if p > 0.05 else 'Not Normal'}")
```

## Hypothesis Testing

Hypothesis testing is a method for making decisions about a population based on sample data.

### One-Sample t-test

Tests if the mean of a sample is significantly different from a known value.

```python
# One-sample t-test
# H0: The mean of the variable equals the hypothesized value
# H1: The mean of the variable does not equal the hypothesized value

variable = 'PRICE'
hypothesized_mean = 20  # Hypothesized value

t_stat, p_value = stats.ttest_1samp(df[variable], hypothesized_mean)
print(f"One-Sample t-test for {variable}:")
print(f"Hypothesized mean: {hypothesized_mean}")
print(f"Sample mean: {df[variable].mean():.4f}")
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
print(f"Conclusion: {'Reject' if p_value < 0.05 else 'Fail to reject'} the null hypothesis (α = 0.05)")
print(f"The mean of {variable} is {'not ' if p_value < 0.05 else ''}equal to {hypothesized_mean}")
```

### Two-Sample t-test

Tests if the means of two independent samples are significantly different.

```python
# Create two groups for comparison
group1 = df[df['RM'] > 6]['PRICE']  # Houses with more than 6 rooms
group2 = df[df['RM'] <= 6]['PRICE']  # Houses with 6 or fewer rooms

# Two-sample t-test
# H0: The means of the two groups are equal
# H1: The means of the two groups are not equal

# First, check if variances are equal
levene_stat, levene_p = stats.levene(group1, group2)
print(f"Levene's test for equal variances:")
print(f"Statistic: {levene_stat:.4f}, p-value: {levene_p:.4f}")
print(f"Variances are {'equal' if levene_p > 0.05 else 'not equal'} (α = 0.05)\n")

# Perform t-test with appropriate equal_var parameter
t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=(levene_p > 0.05))
print(f"Two-Sample t-test for PRICE between houses with >6 rooms and ≤6 rooms:")
print(f"Group 1 mean (>6 rooms): {group1.mean():.4f}")
print(f"Group 2 mean (≤6 rooms): {group2.mean():.4f}")
print(f"Mean difference: {group1.mean() - group2.mean():.4f}")
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
print(f"Conclusion: {'Reject' if p_value < 0.05 else 'Fail to reject'} the null hypothesis (α = 0.05)")
print(f"The means of the two groups are {'not ' if p_value < 0.05 else ''}equal")

# Visualize the comparison
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['RM'] > 6, y=df['PRICE'], data=df)
plt.xlabel('More than 6 rooms')
plt.xticks([0, 1], ['≤6 rooms', '>6 rooms'])
plt.ylabel('Price')
plt.title('House Prices by Number of Rooms')
plt.show()
```

### Paired t-test

Tests if the mean difference between paired observations is significantly different from zero.

```python
# For demonstration, let's create paired data (before/after scenario)
np.random.seed(42)
before = np.random.normal(50, 10, 30)
effect = np.random.normal(5, 2, 30)  # Positive effect with some noise
after = before + effect

# Paired t-test
# H0: The mean difference between paired observations is zero
# H1: The mean difference between paired observations is not zero

t_stat, p_value = stats.ttest_rel(before, after)
print(f"Paired t-test:")
print(f"Mean before: {before.mean():.4f}")
print(f"Mean after: {after.mean():.4f}")
print(f"Mean difference: {(after - before).mean():.4f}")
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
print(f"Conclusion: {'Reject' if p_value < 0.05 else 'Fail to reject'} the null hypothesis (α = 0.05)")
print(f"The mean difference is {'not ' if p_value < 0.05 else ''}zero")

# Visualize the paired data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(before, after)
plt.plot([min(before), max(before)], [min(before), max(before)], 'r--')  # Identity line
plt.xlabel('Before')
plt.ylabel('After')
plt.title('Before vs. After')

plt.subplot(1, 2, 2)
differences = after - before
sns.histplot(differences, kde=True)
plt.axvline(0, color='r', linestyle='--')
plt.xlabel('Difference (After - Before)')
plt.title('Histogram of Differences')

plt.tight_layout()
plt.show()
```

### Chi-Square Test for Independence

Tests if there is a significant association between two categorical variables.

```python
# Create categorical variables for demonstration
df['PRICE_CAT'] = pd.qcut(df['PRICE'], 3, labels=['Low', 'Medium', 'High'])
df['RM_CAT'] = pd.qcut(df['RM'], 3, labels=['Few', 'Average', 'Many'])

# Create contingency table
contingency_table = pd.crosstab(df['PRICE_CAT'], df['RM_CAT'])
print("Contingency Table:")
print(contingency_table)

# Chi-square test
# H0: The two categorical variables are independent
# H1: The two categorical variables are not independent

chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Test for Independence:")
print(f"Chi-square statistic: {chi2_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"Conclusion: {'Reject' if p_value < 0.05 else 'Fail to reject'} the null hypothesis (α = 0.05)")
print(f"The variables are {'not ' if p_value < 0.05 else ''}independent")

# Visualize the relationship
plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Contingency Table: Price Category vs. Room Category')
plt.tight_layout()
plt.show()
```

## Calculating Confidence Intervals

Confidence intervals provide a range of values that is likely to contain the true population parameter.

### Confidence Interval for a Mean

```python
# Calculate 95% confidence interval for a mean
variable = 'PRICE'
data = df[variable]
n = len(data)
mean = data.mean()
std_err = stats.sem(data)  # Standard error of the mean
ci_95 = stats.t.interval(0.95, n-1, loc=mean, scale=std_err)

print(f"95% Confidence Interval for the Mean of {variable}:")
print(f"Sample Mean: {mean:.4f}")
print(f"Standard Error: {std_err:.4f}")
print(f"95% CI: ({ci_95[0]:.4f}, {ci_95[1]:.4f})")

# Visualize the confidence interval
plt.figure(figsize=(10, 6))
sns.histplot(data, kde=True)
plt.axvline(mean, color='red', linestyle='-', label=f'Mean: {mean:.4f}')
plt.axvline(ci_95[0], color='green', linestyle='--', label=f'Lower 95% CI: {ci_95[0]:.4f}')
plt.axvline(ci_95[1], color='green', linestyle='--', label=f'Upper 95% CI: {ci_95[1]:.4f}')
plt.title(f'Distribution of {variable} with 95% Confidence Interval')
plt.legend()
plt.show()
```

### Confidence Interval for a Proportion

```python
# Create a binary variable for demonstration
df['HIGH_PRICE'] = (df['PRICE'] > df['PRICE'].median()).astype(int)
proportion = df['HIGH_PRICE'].mean()
n = len(df)

# Calculate 95% confidence interval for a proportion
from statsmodels.stats.proportion import proportion_confint
ci_95 = proportion_confint(count=df['HIGH_PRICE'].sum(), nobs=n, alpha=0.05, method='normal')

print(f"95% Confidence Interval for the Proportion of High-Priced Houses:")
print(f"Sample Proportion: {proportion:.4f}")
print(f"95% CI: ({ci_95[0]:.4f}, {ci_95[1]:.4f})")

# Visualize the confidence interval
plt.figure(figsize=(10, 6))
sns.countplot(x='HIGH_PRICE', data=df)
plt.axhline(n * proportion, color='red', linestyle='-', label=f'Proportion: {proportion:.4f}')
plt.axhline(n * ci_95[0], color='green', linestyle='--', label=f'Lower 95% CI: {ci_95[0]:.4f}')
plt.axhline(n * ci_95[1], color='green', linestyle='--', label=f'Upper 95% CI: {ci_95[1]:.4f}')
plt.xticks([0, 1], ['Low Price', 'High Price'])
plt.title('Distribution of House Prices with 95% Confidence Interval for Proportion')
plt.legend()
plt.show()
```

### Bootstrap Confidence Intervals

Bootstrap methods are useful when the sampling distribution is unknown or when working with small samples.

```python
# Bootstrap confidence interval for the mean
from sklearn.utils import resample

# Function to calculate bootstrap confidence interval
def bootstrap_ci(data, n_bootstraps=1000, ci=0.95, statistic=np.mean):
    """Calculate bootstrap confidence interval for a statistic."""
    bootstrap_stats = []
    for _ in range(n_bootstraps):
        bootstrap_sample = resample(data, replace=True, n_samples=len(data))
        bootstrap_stats.append(statistic(bootstrap_sample))
    
    # Calculate confidence interval
    alpha = (1 - ci) / 2
    lower_bound = np.percentile(bootstrap_stats, 100 * alpha)
    upper_bound = np.percentile(bootstrap_stats, 100 * (1 - alpha))
    
    return lower_bound, upper_bound

# Calculate bootstrap CI for mean
variable = 'PRICE'
data = df[variable]
mean = data.mean()
bootstrap_ci_mean = bootstrap_ci(data, statistic=np.mean)

print(f"Bootstrap 95% Confidence Interval for the Mean of {variable}:")
print(f"Sample Mean: {mean:.4f}")
print(f"95% Bootstrap CI: ({bootstrap_ci_mean[0]:.4f}, {bootstrap_ci_mean[1]:.4f})")

# Calculate bootstrap CI for median
median = data.median()
bootstrap_ci_median = bootstrap_ci(data, statistic=np.median)

print(f"\nBootstrap 95% Confidence Interval for the Median of {variable}:")
print(f"Sample Median: {median:.4f}")
print(f"95% Bootstrap CI: ({bootstrap_ci_median[0]:.4f}, {bootstrap_ci_median[1]:.4f})")

# Visualize bootstrap distribution for mean
bootstrap_means = [np.mean(resample(data, replace=True, n_samples=len(data))) for _ in range(1000)]

plt.figure(figsize=(10, 6))
sns.histplot(bootstrap_means, kde=True)
plt.axvline(mean, color='red', linestyle='-', label=f'Sample Mean: {mean:.4f}')
plt.axvline(bootstrap_ci_mean[0], color='green', linestyle='--', label=f'Lower 95% CI: {bootstrap_ci_mean[0]:.4f}')
plt.axvline(bootstrap_ci_mean[1], color='green', linestyle='--', label=f'Upper 95% CI: {bootstrap_ci_mean[1]:.4f}')
plt.title(f'Bootstrap Distribution of Mean {variable} with 95% CI')
plt.legend()
plt.show()
```

## Performing ANOVA for Group Comparisons

Analysis of Variance (ANOVA) tests if there are significant differences between the means of three or more independent groups.

### One-Way ANOVA

```python
# Create groups for ANOVA
df['CHAS_CAT'] = df['CHAS'].astype(int).astype(str)
df['RAD_CAT'] = pd.qcut(df['RAD'], 3, labels=['Low', 'Medium', 'High'])

# One-way ANOVA for RAD_CAT
# H0: The means of all groups are equal
# H1: At least one group mean is different

groups = [df[df['RAD_CAT'] == cat]['PRICE'] for cat in df['RAD_CAT'].unique()]
f_stat, p_value = stats.f_oneway(*groups)

print(f"One-Way ANOVA for PRICE across RAD_CAT groups:")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Conclusion: {'Reject' if p_value < 0.05 else 'Fail to reject'} the null hypothesis (α = 0.05)")
print(f"There {'is' if p_value < 0.05 else 'is not'} a significant difference in means across groups")

# Group means
group_means = df.groupby('RAD_CAT')['PRICE'].mean()
print("\nGroup Means:")
print(group_means)

# Visualize the groups
plt.figure(figsize=(10, 6))
sns.boxplot(x='RAD_CAT', y='PRICE', data=df)
plt.title('House Prices by Accessibility to Radial Highways')
plt.xlabel('Accessibility to Radial Highways')
plt.ylabel('Price')
plt.show()
```

### Post-Hoc Tests

If ANOVA indicates significant differences, post-hoc tests help identify which specific groups differ.

```python
# Tukey's HSD (Honest Significant Differences) test
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Perform Tukey's test
tukey = pairwise_tukeyhsd(endog=df['PRICE'], groups=df['RAD_CAT'], alpha=0.05)
print("Tukey's HSD Test Results:")
print(tukey)

# Visualize the results
plt.figure(figsize=(10, 6))
tukey.plot_simultaneous()
plt.title("Tukey's HSD Test for Multiple Comparisons")
plt.tight_layout()
plt.show()
```

### Two-Way ANOVA

Tests the effect of two categorical independent variables on a continuous dependent variable.

```python
# Two-way ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Fit the model
model = ols('PRICE ~ C(RAD_CAT) + C(CHAS_CAT) + C(RAD_CAT):C(CHAS_CAT)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("Two-Way ANOVA Results:")
print(anova_table)

# Interpret the results
for effect in ['C(RAD_CAT)', 'C(CHAS_CAT)', 'C(RAD_CAT):C(CHAS_CAT)']:
    p_value = anova_table.loc[effect, 'PR(>F)']
    print(f"Effect of {effect}: {'Significant' if p_value < 0.05 else 'Not significant'} (p = {p_value:.4f})")

# Visualize the interaction
plt.figure(figsize=(12, 6))
sns.boxplot(x='RAD_CAT', y='PRICE', hue='CHAS_CAT', data=df)
plt.title('House Prices by Accessibility to Highways and Charles River')
plt.xlabel('Accessibility to Radial Highways')
plt.ylabel('Price')
plt.legend(title='Charles River Dummy')
plt.show()
```

## Applying Non-Parametric Tests

Non-parametric tests are useful when data doesn't meet the assumptions of parametric tests (e.g., normality).

### Mann-Whitney U Test

Non-parametric alternative to the independent samples t-test.

```python
# Mann-Whitney U test
# H0: The distributions of both groups are equal
# H1: The distributions of both groups are not equal

group1 = df[df['RM'] > 6]['PRICE']  # Houses with more than 6 rooms
group2 = df[df['RM'] <= 6]['PRICE']  # Houses with 6 or fewer rooms

u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
print(f"Mann-Whitney U Test for PRICE between houses with >6 rooms and ≤6 rooms:")
print(f"U-statistic: {u_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Conclusion: {'Reject' if p_value < 0.05 else 'Fail to reject'} the null hypothesis (α = 0.05)")
print(f"The distributions of the two groups are {'not ' if p_value < 0.05 else ''}equal")

# Compare with t-test results
t_stat, t_p_value = stats.ttest_ind(group1, group2, equal_var=False)
print(f"\nFor comparison, t-test p-value: {t_p_value:.4f}")
```

### Wilcoxon Signed-Rank Test

Non-parametric alternative to the paired samples t-test.

```python
# Wilcoxon signed-rank test
# H0: The median difference between pairs is zero
# H1: The median difference between pairs is not zero

# Using the same paired data as before
w_stat, p_value = stats.wilcoxon(before, after)
print(f"Wilcoxon Signed-Rank Test:")
print(f"W-statistic: {w_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Conclusion: {'Reject' if p_value < 0.05 else 'Fail to reject'} the null hypothesis (α = 0.05)")
print(f"The median difference is {'not ' if p_value < 0.05 else ''}zero")

# Compare with t-test results
t_stat, t_p_value = stats.ttest_rel(before, after)
print(f"\nFor comparison, paired t-test p-value: {t_p_value:.4f}")
```

### Kruskal-Wallis H Test

Non-parametric alternative to one-way ANOVA.

```python
# Kruskal-Wallis H test
# H0: The distributions of all groups are equal
# H1: At least one group distribution is different

groups = [df[df['RAD_CAT'] == cat]['PRICE'] for cat in df['RAD_CAT'].unique()]
h_stat, p_value = stats.kruskal(*groups)

print(f"Kruskal-Wallis H Test for PRICE across RAD_CAT groups:")
print(f"H-statistic: {h_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Conclusion: {'Reject' if p_value < 0.05 else 'Fail to reject'} the null hypothesis (α = 0.05)")
print(f"There {'is' if p_value < 0.05 else 'is not'} a significant difference in distributions across groups")

# Compare with ANOVA results
f_stat, f_p_value = stats.f_oneway(*groups)
print(f"\nFor comparison, ANOVA p-value: {f_p_value:.4f}")
```

## Comprehensive Statistical Analysis Example

Let's put everything together with a comprehensive statistical analysis on a real dataset.

```python
# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

print("Dataset Overview:")
print(titanic.head())
print(f"\nShape: {titanic.shape}")

# 1. Descriptive Statistics
print("\nDescriptive Statistics for Numeric Variables:")
print(titanic.describe())

# 2. Check Normality of Age
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(titanic['age'].dropna(), kde=True)
plt.title('Distribution of Age')

plt.subplot(1, 2, 2)
stats.probplot(titanic['age'].dropna(), plot=plt)
plt.title('Q-Q Plot of Age')

plt.tight_layout()
plt.show()

# Shapiro-Wilk test for normality
stat, p_value = stats.shapiro(titanic['age'].dropna())
print(f"\nShapiro-Wilk Test for Age:")
print(f"Statistic: {stat:.4f}, p-value: {p_value:.4f}")
print(f"Conclusion: Age {'is' if p_value > 0.05 else 'is not'} normally distributed (α = 0.05)")

# 3. Compare Age between Survivors and Non-Survivors
survivors = titanic[titanic['survived'] == 1]['age'].dropna()
non_survivors = titanic[titanic['survived'] == 0]['age'].dropna()

# Visualize the comparison
plt.figure(figsize=(10, 6))
sns.boxplot(x='survived', y='age', data=titanic)
plt.title('Age by Survival Status')
plt.xlabel('Survived')
plt.xticks([0, 1], ['No', 'Yes'])
plt.ylabel('Age')
plt.show()

# Check if variances are equal
levene_stat, levene_p = stats.levene(survivors, non_survivors)
print(f"\nLevene's test for equal variances:")
print(f"Statistic: {levene_stat:.4f}, p-value: {levene_p:.4f}")
print(f"Variances are {'equal' if levene_p > 0.05 else 'not equal'} (α = 0.05)")

# Perform t-test
t_stat, p_value = stats.ttest_ind(survivors, non_survivors, equal_var=(levene_p > 0.05))
print(f"\nIndependent Samples t-test for Age between Survivors and Non-Survivors:")
print(f"Survivors mean age: {survivors.mean():.2f}")
print(f"Non-survivors mean age: {non_survivors.mean():.2f}")
print(f"Mean difference: {survivors.mean() - non_survivors.mean():.2f}")
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
print(f"Conclusion: {'Reject' if p_value < 0.05 else 'Fail to reject'} the null hypothesis (α = 0.05)")
print(f"The mean ages {'are' if p_value < 0.05 else 'are not'} significantly different")

# 4. Chi-Square Test for Independence between Sex and Survival
contingency_table = pd.crosstab(titanic['sex'], titanic['survived'])
print("\nContingency Table (Sex vs. Survival):")
print(contingency_table)

chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Test for Independence between Sex and Survival:")
print(f"Chi-square statistic: {chi2_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"Conclusion: {'Reject' if p_value < 0.05 else 'Fail to reject'} the null hypothesis (α = 0.05)")
print(f"Sex and survival {'are' if p_value < 0.05 else 'are not'} dependent")

# Visualize the relationship
plt.figure(figsize=(10, 6))
sns.countplot(x='sex', hue='survived', data=titanic)
plt.title('Survival by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# 5. ANOVA: Compare Fare across Passenger Classes
classes = [titanic[titanic['pclass'] == cls]['fare'].dropna() for cls in [1, 2, 3]]
f_stat, p_value = stats.f_oneway(*classes)

print(f"\nOne-Way ANOVA for Fare across Passenger Classes:")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Conclusion: {'Reject' if p_value < 0.05 else 'Fail to reject'} the null hypothesis (α = 0.05)")
print(f"There {'is' if p_value < 0.05 else 'is not'} a significant difference in fares across classes")

# Visualize the comparison
plt.figure(figsize=(10, 6))
sns.boxplot(x='pclass', y='fare', data=titanic)
plt.title('Fare by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.show()

# 6. Post-hoc test for ANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Prepare data for Tukey's test
class_data = titanic[['pclass', 'fare']].dropna()
tukey = pairwise_tukeyhsd(endog=class_data['fare'], groups=class_data['pclass'], alpha=0.05)
print("\nTukey's HSD Test Results for Fare across Passenger Classes:")
print(tukey)

# 7. Confidence Interval for Survival Rate
survival_rate = titanic['survived'].mean()
n = len(titanic)
ci_95 = proportion_confint(count=titanic['survived'].sum(), nobs=n, alpha=0.05, method='normal')

print(f"\n95% Confidence Interval for the Survival Rate:")
print(f"Sample Proportion: {survival_rate:.4f}")
print(f"95% CI: ({ci_95[0]:.4f}, {ci_95[1]:.4f})")

# 8. Summary of Findings
print("\nKey Statistical Findings:")
print("1. Age distribution is not normally distributed")
print("2. There is a significant difference in age between survivors and non-survivors")
print("3. Sex and survival are significantly dependent, with females having higher survival rates")
print("4. There is a significant difference in fares across passenger classes")
print("5. All pairwise comparisons between passenger classes show significant differences in fares")
print(f"6. The overall survival rate was {survival_rate:.2%} (95% CI: {ci_95[0]:.2%} to {ci_95[1]:.2%})")
```

## Practice Exercises

1. Test a dataset for normality using both visual methods and statistical tests.
2. Apply appropriate transformations to make a non-normal dataset more normal.
3. Perform a t-test to compare means between two groups in a dataset.
4. Calculate and interpret confidence intervals for means and proportions.
5. Conduct a chi-square test to examine the relationship between two categorical variables.
6. Perform ANOVA to compare means across multiple groups, followed by post-hoc tests.
7. Apply non-parametric tests when data doesn't meet parametric assumptions.
8. Conduct a comprehensive statistical analysis on a dataset of your choice, including hypothesis testing and confidence intervals.

## Key Takeaways

- Testing for normality is an important first step in statistical analysis
- Transformations can help make non-normal data more suitable for parametric tests
- Hypothesis testing allows us to make inferences about populations based on sample data
- Confidence intervals quantify the uncertainty in our estimates
- ANOVA and post-hoc tests help compare means across multiple groups
- Non-parametric tests provide alternatives when parametric assumptions are not met
- Statistical analysis should be guided by clear research questions and appropriate methods
- Visualization is an essential complement to statistical tests for understanding and communicating results
