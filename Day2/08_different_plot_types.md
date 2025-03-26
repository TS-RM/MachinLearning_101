# Creating Different Types of Plots and Charts

## Introduction to Plot Types

Data visualization is a powerful way to communicate insights and patterns in your data. Different types of plots serve different purposes, and choosing the right visualization is crucial for effective data communication. This guide covers various plot types beyond the basics, including time series visualizations, stacked and grouped bar charts, pie charts, heatmaps, and geographic visualizations.

## Time Series Visualizations

Time series visualizations display data points indexed in time order, allowing you to observe trends, seasonality, and patterns over time.

### Basic Line Plot for Time Series

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample time series data
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
values = np.cumsum(np.random.randn(365)) + 100  # Random walk with starting value of 100
ts_df = pd.DataFrame({'date': dates, 'value': values})

# Create a basic time series plot
plt.figure(figsize=(12, 6))
plt.plot(ts_df['date'], ts_df['value'])
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Multiple Time Series

```python
# Create multiple time series
ts_df['value2'] = np.cumsum(np.random.randn(365)) + 120
ts_df['value3'] = np.cumsum(np.random.randn(365)) + 80

# Plot multiple time series
plt.figure(figsize=(12, 6))
plt.plot(ts_df['date'], ts_df['value'], label='Series 1')
plt.plot(ts_df['date'], ts_df['value2'], label='Series 2')
plt.plot(ts_df['date'], ts_df['value3'], label='Series 3')
plt.title('Multiple Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Time Series with Seaborn

```python
# Melt the dataframe for Seaborn
ts_melted = pd.melt(ts_df, id_vars=['date'], value_vars=['value', 'value2', 'value3'],
                    var_name='series', value_name='value')

# Create a time series plot with Seaborn
plt.figure(figsize=(12, 6))
sns.lineplot(data=ts_melted, x='date', y='value', hue='series')
plt.title('Time Series with Seaborn')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Area Plot for Time Series

```python
# Create an area plot
plt.figure(figsize=(12, 6))
plt.fill_between(ts_df['date'], ts_df['value'], alpha=0.5, label='Series 1')
plt.fill_between(ts_df['date'], ts_df['value2'], alpha=0.5, label='Series 2')
plt.fill_between(ts_df['date'], ts_df['value3'], alpha=0.5, label='Series 3')
plt.title('Area Plot for Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Stacked Area Plot

```python
# Create a stacked area plot
plt.figure(figsize=(12, 6))
plt.stackplot(ts_df['date'], ts_df['value'], ts_df['value2'], ts_df['value3'],
              labels=['Series 1', 'Series 2', 'Series 3'], alpha=0.7)
plt.title('Stacked Area Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Time Series with Moving Averages

```python
# Calculate moving averages
ts_df['MA7'] = ts_df['value'].rolling(window=7).mean()  # 7-day moving average
ts_df['MA30'] = ts_df['value'].rolling(window=30).mean()  # 30-day moving average

# Plot time series with moving averages
plt.figure(figsize=(12, 6))
plt.plot(ts_df['date'], ts_df['value'], alpha=0.5, label='Original')
plt.plot(ts_df['date'], ts_df['MA7'], label='7-day MA')
plt.plot(ts_df['date'], ts_df['MA30'], label='30-day MA')
plt.title('Time Series with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Stacked and Grouped Bar Charts

Bar charts are excellent for comparing categorical data. Stacked and grouped bar charts allow you to compare multiple categories and subcategories.

### Grouped Bar Chart

```python
# Create sample data for grouped bar chart
categories = ['Category A', 'Category B', 'Category C', 'Category D']
values1 = [15, 30, 45, 22]
values2 = [25, 18, 32, 41]
values3 = [20, 25, 30, 35]

# Set up the figure
fig, ax = plt.subplots(figsize=(12, 6))

# Set width of bars
barWidth = 0.25

# Set positions of the bars on X axis
r1 = np.arange(len(categories))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Create bars
ax.bar(r1, values1, width=barWidth, label='Group 1', color='skyblue')
ax.bar(r2, values2, width=barWidth, label='Group 2', color='lightgreen')
ax.bar(r3, values3, width=barWidth, label='Group 3', color='salmon')

# Add labels and title
ax.set_xlabel('Categories')
ax.set_ylabel('Values')
ax.set_title('Grouped Bar Chart')
ax.set_xticks([r + barWidth for r in range(len(categories))])
ax.set_xticklabels(categories)
ax.legend()

plt.tight_layout()
plt.show()
```

### Grouped Bar Chart with Pandas

```python
# Create a DataFrame for grouped bar chart
grouped_df = pd.DataFrame({
    'Category': categories * 3,
    'Group': ['Group 1'] * 4 + ['Group 2'] * 4 + ['Group 3'] * 4,
    'Value': values1 + values2 + values3
})

# Create grouped bar chart with pandas
grouped_pivot = grouped_df.pivot(index='Category', columns='Group', values='Value')
grouped_pivot.plot(kind='bar', figsize=(12, 6))
plt.title('Grouped Bar Chart with Pandas')
plt.xlabel('Category')
plt.ylabel('Value')
plt.tight_layout()
plt.show()
```

### Stacked Bar Chart

```python
# Create a stacked bar chart
fig, ax = plt.subplots(figsize=(12, 6))

# Create bars
ax.bar(categories, values1, label='Group 1', color='skyblue')
ax.bar(categories, values2, bottom=values1, label='Group 2', color='lightgreen')
ax.bar(categories, values3, bottom=[i+j for i,j in zip(values1, values2)], label='Group 3', color='salmon')

# Add labels and title
ax.set_xlabel('Categories')
ax.set_ylabel('Values')
ax.set_title('Stacked Bar Chart')
ax.legend()

plt.tight_layout()
plt.show()
```

### Stacked Bar Chart with Pandas

```python
# Create stacked bar chart with pandas
grouped_pivot.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Stacked Bar Chart with Pandas')
plt.xlabel('Category')
plt.ylabel('Value')
plt.tight_layout()
plt.show()
```

### 100% Stacked Bar Chart

```python
# Calculate percentages for 100% stacked bar chart
total = [i+j+k for i,j,k in zip(values1, values2, values3)]
percent1 = [i/t*100 for i,t in zip(values1, total)]
percent2 = [i/t*100 for i,t in zip(values2, total)]
percent3 = [i/t*100 for i,t in zip(values3, total)]

# Create 100% stacked bar chart
fig, ax = plt.subplots(figsize=(12, 6))

# Create bars
ax.bar(categories, percent1, label='Group 1', color='skyblue')
ax.bar(categories, percent2, bottom=percent1, label='Group 2', color='lightgreen')
ax.bar(categories, percent3, bottom=[i+j for i,j in zip(percent1, percent2)], label='Group 3', color='salmon')

# Add labels and title
ax.set_xlabel('Categories')
ax.set_ylabel('Percentage (%)')
ax.set_title('100% Stacked Bar Chart')
ax.legend()

plt.tight_layout()
plt.show()
```

## Pie and Donut Charts

Pie charts display data as slices of a circle, showing the proportion of each category to the whole.

### Basic Pie Chart

```python
# Create sample data for pie chart
labels = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
sizes = [15, 30, 25, 10, 20]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen']
explode = (0.1, 0, 0, 0, 0)  # explode 1st slice

# Create pie chart
plt.figure(figsize=(10, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.title('Pie Chart')
plt.tight_layout()
plt.show()
```

### Donut Chart

```python
# Create a donut chart (pie chart with a hole in the middle)
plt.figure(figsize=(10, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140, wedgeprops=dict(width=0.5))  # width controls the size of the hole
plt.axis('equal')
plt.title('Donut Chart')
plt.tight_layout()
plt.show()
```

### Nested Pie Chart

```python
# Create data for nested pie chart
outer_sizes = [40, 60]
outer_labels = ['Group 1', 'Group 2']
outer_colors = ['lightblue', 'lightgreen']

inner_sizes = [15, 25, 35, 25]
inner_labels = ['A', 'B', 'C', 'D']
inner_colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

# Create nested pie chart
plt.figure(figsize=(10, 8))

# Outer pie chart
plt.pie(outer_sizes, labels=outer_labels, colors=outer_colors, autopct='%1.1f%%',
        startangle=90, radius=1, wedgeprops=dict(width=0.3, edgecolor='w'))

# Inner pie chart
plt.pie(inner_sizes, labels=inner_labels, colors=inner_colors, autopct='%1.1f%%',
        startangle=90, radius=0.7, wedgeprops=dict(width=0.4, edgecolor='w'))

# Add a circle at the center to create a donut chart
centre_circle = plt.Circle((0,0), 0.3, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.axis('equal')
plt.title('Nested Pie Chart')
plt.tight_layout()
plt.show()
```

## Heatmaps and Correlation Matrices

Heatmaps display values in a matrix as colors, making it easy to visualize patterns and relationships.

### Basic Heatmap

```python
# Create sample data for heatmap
data = np.random.rand(10, 12)
rows = ['Row ' + str(i) for i in range(1, 11)]
columns = ['Col ' + str(i) for i in range(1, 13)]

# Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data, annot=True, fmt='.2f', cmap='viridis', xticklabels=columns, yticklabels=rows)
plt.title('Basic Heatmap')
plt.tight_layout()
plt.show()
```

### Correlation Matrix Heatmap

```python
# Create sample data for correlation matrix
np.random.seed(42)
df = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
df['B'] = df['A'] + np.random.randn(100) * 0.5  # B is correlated with A
df['D'] = -df['C'] + np.random.randn(100) * 0.5  # D is negatively correlated with C

# Calculate correlation matrix
corr = df.corr()

# Create correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={'shrink': .8})
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()
```

### Heatmap with Mask

```python
# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Create heatmap with mask
plt.figure(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={'shrink': .8})
plt.title('Correlation Matrix Heatmap (Lower Triangle)')
plt.tight_layout()
plt.show()
```

### Clustered Heatmap

```python
# Create clustered heatmap
plt.figure(figsize=(12, 10))
clustered_heatmap = sns.clustermap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, center=0,
                                  linewidths=.5, figsize=(12, 10))
plt.title('Clustered Heatmap')
plt.tight_layout()
plt.show()
```

## Geographic Visualizations with Maps

Geographic visualizations display data on maps, allowing you to see spatial patterns and relationships.

### Basic Map with Geopandas

```python
# Install required packages if not already installed
# !pip install geopandas matplotlib

import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap

# Load world map data
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Create a basic world map
fig, ax = plt.subplots(figsize=(15, 10))
world.plot(ax=ax, color='lightgrey', edgecolor='white')
ax.set_title('World Map')
plt.tight_layout()
plt.show()
```

### Choropleth Map

```python
# Create sample data for countries
np.random.seed(42)
world['value'] = np.random.randint(1, 100, size=len(world))

# Create a choropleth map
fig, ax = plt.subplots(figsize=(15, 10))
world.plot(column='value', ax=ax, legend=True, cmap='YlOrRd', 
           legend_kwds={'label': "Value", 'orientation': "horizontal"})
ax.set_title('Choropleth Map')
plt.tight_layout()
plt.show()
```

### Bubble Map

```python
# Create sample data for cities
cities = pd.DataFrame({
    'name': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney', 'Beijing', 'Rio de Janeiro'],
    'lon': [-74.0060, -0.1278, 139.6917, 2.3522, 151.2093, 116.4074, -43.1729],
    'lat': [40.7128, 51.5074, 35.6895, 48.8566, -33.8688, 39.9042, -22.9068],
    'population': [8.4, 8.9, 13.9, 2.1, 5.3, 21.5, 6.7]  # in millions
})

# Convert to GeoDataFrame
from shapely.geometry import Point
geometry = [Point(xy) for xy in zip(cities['lon'], cities['lat'])]
cities_gdf = gpd.GeoDataFrame(cities, geometry=geometry)

# Create a bubble map
fig, ax = plt.subplots(figsize=(15, 10))
world.plot(ax=ax, color='lightgrey', edgecolor='white')
cities_gdf.plot(ax=ax, markersize=cities_gdf['population']*20, color='red', alpha=0.7)

# Add city labels
for idx, row in cities_gdf.iterrows():
    ax.annotate(row['name'], xy=(row.geometry.x, row.geometry.y), xytext=(5, 5), 
                textcoords="offset points", fontsize=10)

ax.set_title('Bubble Map of Major Cities')
plt.tight_layout()
plt.show()
```

### Interactive Maps with Folium

```python
# Install required packages if not already installed
# !pip install folium

import folium

# Create a base map
m = folium.Map(location=[0, 0], zoom_start=2)

# Add markers for cities
for idx, row in cities.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=row['population'],
        popup=row['name'] + ': ' + str(row['population']) + ' million',
        color='red',
        fill=True,
        fill_color='red'
    ).add_to(m)

# Save the map
m.save('interactive_map.html')

# Display the map (in Jupyter notebook)
# m
```

## Practice Exercises

1. Create a time series visualization showing stock prices for multiple companies over a year.
2. Build a stacked bar chart showing sales by product category and region.
3. Create a grouped bar chart comparing quarterly performance across different years.
4. Design a pie chart and a donut chart to represent market share data.
5. Generate a heatmap showing correlation between different variables in a dataset.
6. Create a choropleth map showing population density by country or state.
7. Build an interactive map with markers for points of interest.
8. Create a dashboard with multiple plot types showing different aspects of the same dataset.

## Key Takeaways

- Time series visualizations help identify trends, seasonality, and patterns over time
- Stacked and grouped bar charts allow comparison of multiple categories and subcategories
- Pie and donut charts show proportions of a whole, but should be used sparingly and with few categories
- Heatmaps and correlation matrices help visualize relationships between variables
- Geographic visualizations display data on maps, revealing spatial patterns
- Different plot types serve different purposes; choose the right visualization for your data and message
- Combining multiple plot types can provide a more comprehensive view of your data
