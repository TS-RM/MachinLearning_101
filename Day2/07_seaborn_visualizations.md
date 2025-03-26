# Seaborn for Statistical Visualizations

## Introduction to Seaborn

Seaborn is a Python data visualization library based on Matplotlib that provides a high-level interface for creating attractive and informative statistical graphics. It's designed to work well with pandas DataFrames and integrates closely with the PyData ecosystem. Seaborn simplifies the process of creating complex visualizations while maintaining the flexibility of Matplotlib.

## Understanding Seaborn's Relationship with Matplotlib

Seaborn builds on top of Matplotlib, providing:

1. **Higher-level abstractions**: Seaborn functions typically require less code than equivalent Matplotlib code
2. **Attractive default styles**: Seaborn comes with aesthetically pleasing default themes
3. **Built-in statistical functionality**: Many Seaborn plots automatically calculate and display statistical information
4. **DataFrame integration**: Seaborn works seamlessly with pandas DataFrames

Seaborn doesn't replace Matplotlib; it enhances it. You can use both libraries together, customizing Seaborn plots with Matplotlib functions.

### Installing Seaborn

```python
pip install seaborn
```

To verify your installation:

```python
import seaborn as sns
print(sns.__version__)
```

### Setting Up Seaborn

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style
sns.set_style("whitegrid")  # Options: darkgrid, whitegrid, dark, white, ticks

# Set the context (controls the scaling of plot elements)
sns.set_context("notebook")  # Options: paper, notebook, talk, poster

# Set the color palette
sns.set_palette("deep")  # Options: deep, muted, pastel, bright, dark, colorblind
```

## Distribution Plots

Distribution plots help visualize the distribution of a dataset. Seaborn provides several functions for this purpose.

### Histplot

Histograms show the distribution of a single variable by dividing the x-axis into bins and counting the number of observations in each bin.

```python
# Create sample data
data = np.random.normal(0, 1, 1000)

# Create a histogram
plt.figure(figsize=(10, 6))
sns.histplot(data, bins=30, kde=True)
plt.title('Histogram with KDE')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Histogram with multiple distributions
df = pd.DataFrame({
    'Group A': np.random.normal(0, 1, 1000),
    'Group B': np.random.normal(2, 1.5, 1000)
})

plt.figure(figsize=(10, 6))
sns.histplot(df, bins=30, kde=True)
plt.title('Multiple Distributions')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

### KDEplot

Kernel Density Estimation (KDE) plots show a smooth curve representing the distribution.

```python
# Create a KDE plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data, fill=True)
plt.title('KDE Plot')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()

# Multiple KDE plots
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, fill=True, common_norm=False, alpha=0.5)
plt.title('Multiple KDE Plots')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()
```

### Rugplot

Rugplots show the distribution of data points along an axis.

```python
# Create a rugplot
plt.figure(figsize=(10, 6))
sns.rugplot(data)
plt.title('Rugplot')
plt.xlabel('Value')
plt.show()

# Combine rugplot with KDE
plt.figure(figsize=(10, 6))
sns.kdeplot(data, fill=True)
sns.rugplot(data, color='red')
plt.title('KDE with Rugplot')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()
```

### Combining Distribution Plots

```python
# Create a figure with multiple distribution plots
plt.figure(figsize=(12, 8))

# Create a subplot grid
gs = plt.GridSpec(2, 2)

# Histogram
ax1 = plt.subplot(gs[0, 0])
sns.histplot(data, bins=30, ax=ax1)
ax1.set_title('Histogram')

# KDE plot
ax2 = plt.subplot(gs[0, 1])
sns.kdeplot(data, fill=True, ax=ax2)
ax2.set_title('KDE Plot')

# Rugplot
ax3 = plt.subplot(gs[1, 0])
sns.rugplot(data, ax=ax3)
ax3.set_title('Rugplot')

# Combined plot
ax4 = plt.subplot(gs[1, 1])
sns.histplot(data, bins=30, kde=True, ax=ax4)
ax4.set_title('Histogram with KDE')

plt.tight_layout()
plt.show()
```

## Relationship Plots

Relationship plots help visualize the relationship between two or more variables.

### Scatterplot

Scatterplots show the relationship between two continuous variables.

```python
# Create sample data
np.random.seed(42)
x = np.random.normal(0, 1, 100)
y = x * 2 + np.random.normal(0, 2, 100)
df = pd.DataFrame({'x': x, 'y': y})

# Create a scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='x', y='y', data=df)
plt.title('Scatterplot')
plt.xlabel('X Variable')
plt.ylabel('Y Variable')
plt.show()

# Scatterplot with hue (color) and size
df['category'] = np.random.choice(['A', 'B', 'C'], 100)
df['size'] = np.random.uniform(10, 100, 100)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='x', y='y', hue='category', size='size', data=df, sizes=(20, 200), alpha=0.7)
plt.title('Scatterplot with Hue and Size')
plt.xlabel('X Variable')
plt.ylabel('Y Variable')
plt.show()
```

### Lineplot

Lineplots show the relationship between two variables connected by lines.

```python
# Create time series data
dates = pd.date_range('2023-01-01', periods=30)
df_time = pd.DataFrame({
    'date': dates,
    'value_A': np.cumsum(np.random.randn(30)),
    'value_B': np.cumsum(np.random.randn(30))
})
df_time = pd.melt(df_time, id_vars=['date'], value_vars=['value_A', 'value_B'], var_name='series', value_name='value')

# Create a lineplot
plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='value', hue='series', data=df_time)
plt.title('Line Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Lineplot with confidence intervals
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.2, 100)
df_sin = pd.DataFrame({'x': x, 'y': y})

plt.figure(figsize=(10, 6))
sns.lineplot(x='x', y='y', data=df_sin, ci=95)  # 95% confidence interval
plt.title('Line Plot with Confidence Interval')
plt.xlabel('X Variable')
plt.ylabel('Y Variable')
plt.show()
```

### Regplot

Regplots show a scatter plot with a regression line.

```python
# Create a regplot
plt.figure(figsize=(10, 6))
sns.regplot(x='x', y='y', data=df)
plt.title('Regression Plot')
plt.xlabel('X Variable')
plt.ylabel('Y Variable')
plt.show()

# Regplot with polynomial fit
plt.figure(figsize=(10, 6))
sns.regplot(x='x', y='y', data=df, order=2)  # Quadratic fit
plt.title('Regression Plot with Polynomial Fit')
plt.xlabel('X Variable')
plt.ylabel('Y Variable')
plt.show()
```

## Categorical Plots

Categorical plots help visualize the distribution of a variable across different categories.

### Barplot

Barplots show the relationship between a categorical and a continuous variable.

```python
# Create sample data
categories = ['A', 'B', 'C', 'D', 'E']
values = [10, 24, 17, 32, 28]
errors = [2, 3, 4, 2, 5]

df_bar = pd.DataFrame({'category': categories, 'value': values})

# Create a barplot
plt.figure(figsize=(10, 6))
sns.barplot(x='category', y='value', data=df_bar)
plt.title('Bar Plot')
plt.xlabel('Category')
plt.ylabel('Value')
plt.show()

# Barplot with error bars
plt.figure(figsize=(10, 6))
sns.barplot(x=categories, y=values, yerr=errors)
plt.title('Bar Plot with Error Bars')
plt.xlabel('Category')
plt.ylabel('Value')
plt.show()

# Grouped barplot
df_grouped = pd.DataFrame({
    'category': np.repeat(['A', 'B', 'C', 'D'], 3),
    'group': np.tile(['Group 1', 'Group 2', 'Group 3'], 4),
    'value': np.random.randint(10, 40, 12)
})

plt.figure(figsize=(12, 6))
sns.barplot(x='category', y='value', hue='group', data=df_grouped)
plt.title('Grouped Bar Plot')
plt.xlabel('Category')
plt.ylabel('Value')
plt.legend(title='Group')
plt.show()
```

### Boxplot

Boxplots show the distribution of a continuous variable across different categories.

```python
# Create sample data
df_box = pd.DataFrame({
    'category': np.repeat(['A', 'B', 'C', 'D'], 30),
    'value': np.concatenate([
        np.random.normal(0, 1, 30),
        np.random.normal(2, 1.5, 30),
        np.random.normal(4, 1, 30),
        np.random.normal(2.5, 2, 30)
    ])
})

# Create a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='category', y='value', data=df_box)
plt.title('Box Plot')
plt.xlabel('Category')
plt.ylabel('Value')
plt.show()

# Horizontal boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(y='category', x='value', data=df_box)
plt.title('Horizontal Box Plot')
plt.ylabel('Category')
plt.xlabel('Value')
plt.show()

# Boxplot with individual points
plt.figure(figsize=(10, 6))
sns.boxplot(x='category', y='value', data=df_box)
sns.stripplot(x='category', y='value', data=df_box, color='black', alpha=0.5, jitter=True)
plt.title('Box Plot with Individual Points')
plt.xlabel('Category')
plt.ylabel('Value')
plt.show()
```

### Violinplot

Violinplots combine boxplot and KDE to show the distribution of data across categories.

```python
# Create a violinplot
plt.figure(figsize=(10, 6))
sns.violinplot(x='category', y='value', data=df_box)
plt.title('Violin Plot')
plt.xlabel('Category')
plt.ylabel('Value')
plt.show()

# Split violinplot
df_box['group'] = np.random.choice(['Group 1', 'Group 2'], len(df_box))

plt.figure(figsize=(10, 6))
sns.violinplot(x='category', y='value', hue='group', data=df_box, split=True)
plt.title('Split Violin Plot')
plt.xlabel('Category')
plt.ylabel('Value')
plt.legend(title='Group')
plt.show()

# Violinplot with inner points
plt.figure(figsize=(10, 6))
sns.violinplot(x='category', y='value', data=df_box, inner='points')
plt.title('Violin Plot with Inner Points')
plt.xlabel('Category')
plt.ylabel('Value')
plt.show()
```

## Advanced Features

Seaborn provides several advanced features for more complex visualizations.

### FacetGrid

FacetGrid allows you to create multiple plots based on different subsets of your data.

```python
# Create sample data
tips = sns.load_dataset('tips')

# Create a FacetGrid
g = sns.FacetGrid(tips, col='time', row='sex', height=4, aspect=1.5)
g.map_dataframe(sns.scatterplot, x='total_bill', y='tip')
g.add_legend()
g.fig.suptitle('Tips by Time and Gender', y=1.05)
plt.show()

# FacetGrid with different plots
g = sns.FacetGrid(tips, col='day', height=4, aspect=1)
g.map_dataframe(sns.histplot, x='total_bill')
g.set_axis_labels('Total Bill', 'Count')
g.set_titles('{col_name}')
g.fig.suptitle('Distribution of Total Bill by Day', y=1.02)
plt.show()
```

### PairGrid

PairGrid creates a grid of plots showing pairwise relationships in a dataset.

```python
# Create a PairGrid
iris = sns.load_dataset('iris')

g = sns.PairGrid(iris, hue='species')
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()
plt.show()

# Customize PairGrid
g = sns.PairGrid(iris, hue='species', diag_sharey=False)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.histplot, kde=True)
g.add_legend()
plt.show()
```

### Heatmap

Heatmaps are useful for visualizing matrices, such as correlation matrices.

```python
# Create a correlation matrix
corr = iris.drop('species', axis=1).corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={'shrink': .8})
plt.title('Correlation Matrix')
plt.show()

# Heatmap with mask (to show only half of the matrix)
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={'shrink': .8})
plt.title('Correlation Matrix (Lower Triangle)')
plt.show()
```

## Customizing Seaborn Plots

Seaborn plots can be customized using both Seaborn and Matplotlib functions.

### Themes and Styles

```python
# Available styles
styles = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']

# Create a figure with different styles
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

for i, style in enumerate(styles):
    if i < len(axes):
        with sns.axes_style(style):
            ax = axes[i]
            sns.lineplot(x='total_bill', y='tip', data=tips, ax=ax)
            ax.set_title(f'Style: {style}')

# Remove empty subplot
if len(styles) < len(axes):
    fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()
```

### Color Palettes

```python
# Available palettes
palettes = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']

# Create a figure with different palettes
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

for i, palette in enumerate(palettes):
    if i < len(axes):
        ax = axes[i]
        sns.barplot(x='day', y='total_bill', hue='sex', data=tips, palette=palette, ax=ax)
        ax.set_title(f'Palette: {palette}')
        ax.legend(title='Sex')

plt.tight_layout()
plt.show()

# Custom color palette
custom_palette = sns.color_palette("husl", 8)
plt.figure(figsize=(10, 6))
sns.palplot(custom_palette)
plt.title('Custom Color Palette')
plt.show()
```

### Combining with Matplotlib

```python
# Create a Seaborn plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_bill', y='tip', hue='day', size='size', data=tips)

# Add Matplotlib customizations
plt.title('Tips vs Total Bill', fontsize=16, fontweight='bold')
plt.xlabel('Total Bill ($)', fontsize=12)
plt.ylabel('Tip ($)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(y=tips['tip'].mean(), color='r', linestyle='--', label='Average Tip')
plt.legend(title='Day', title_fontsize=12)

# Add text annotation
avg_tip = tips['tip'].mean()
plt.text(40, avg_tip + 0.5, f'Average Tip: ${avg_tip:.2f}', fontsize=10,
         bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.show()
```

## Practice Exercises

1. Load a dataset (e.g., tips, iris, or titanic from Seaborn's built-in datasets) and create histograms for numeric variables.
2. Create a KDE plot showing the distribution of a variable across different categories.
3. Create a scatterplot showing the relationship between two variables, with points colored by a third variable.
4. Create a lineplot showing a time series with confidence intervals.
5. Create a barplot comparing values across categories, with error bars.
6. Create a boxplot and violinplot side by side for the same data.
7. Use FacetGrid to create multiple plots based on different subsets of your data.
8. Create a heatmap showing the correlation matrix of a dataset.
9. Customize a Seaborn plot with different styles, color palettes, and Matplotlib additions.
10. Create a complex visualization combining multiple Seaborn plot types.

## Key Takeaways

- Seaborn builds on Matplotlib to provide higher-level abstractions for statistical visualizations
- Distribution plots (histplot, kdeplot, rugplot) help visualize the distribution of data
- Relationship plots (scatterplot, lineplot, regplot) show the relationship between variables
- Categorical plots (barplot, boxplot, violinplot) visualize data across categories
- Advanced features like FacetGrid, PairGrid, and heatmap enable more complex visualizations
- Seaborn plots can be customized with different styles, color palettes, and Matplotlib functions
- Seaborn integrates well with pandas DataFrames, making it easy to visualize structured data
