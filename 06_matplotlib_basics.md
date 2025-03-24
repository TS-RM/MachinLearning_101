# Matplotlib Basics

## Introduction to Matplotlib

Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. It provides an object-oriented API for embedding plots into applications and a pyplot interface for interactive plotting. Matplotlib is the foundation of Python's visualization landscape and is used by many other libraries, including Seaborn and Pandas.

## Installing Matplotlib

Before you can use Matplotlib, you need to install it:

```python
pip install matplotlib
```

To verify your installation, run:

```python
import matplotlib
print(matplotlib.__version__)
```

## Understanding the Figure and Axes Objects

Matplotlib's architecture is built around two main objects:

1. **Figure**: The overall window or page that contains everything
2. **Axes**: An individual plot (with its own coordinate system)

A Figure can contain multiple Axes (subplots), and each Axes has methods for plotting various types of data.

### Creating a Figure and Axes

```python
import matplotlib.pyplot as plt
import numpy as np

# Create a Figure and an Axes
fig, ax = plt.subplots()

# Plot some data
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y)

# Add labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('A Simple Sine Wave')

# Display the plot
plt.show()
```

### Understanding the Object-Oriented vs. Pyplot Interfaces

Matplotlib offers two interfaces:

1. **Object-Oriented (OO) Interface**: More flexible and powerful, recommended for complex plots
2. **Pyplot Interface**: More concise, good for simple plots

Here's the same plot using both interfaces:

```python
# Object-Oriented Interface
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('OO Interface')

# Pyplot Interface
plt.figure()
plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Pyplot Interface')
plt.show()
```

For most applications, the object-oriented interface is recommended because it gives you more control and is more suitable for complex visualizations.

## Basic Plot Types

Matplotlib can create a wide variety of plot types. Here are some of the most common ones:

### Line Plot

```python
# Create data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create a figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines
ax.plot(x, y1, label='Sine')
ax.plot(x, y2, label='Cosine')

# Add labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Line Plot Example')
ax.legend()

plt.show()
```

### Scatter Plot

```python
# Create random data
np.random.seed(42)
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)
sizes = 1000 * np.random.rand(50)

# Create a scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(x, y, c=colors, s=sizes, alpha=0.5)

# Add a colorbar
cbar = fig.colorbar(scatter)
cbar.set_label('Color Value')

# Add labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Scatter Plot Example')

plt.show()
```

### Bar Plot

```python
# Create data
categories = ['Category A', 'Category B', 'Category C', 'Category D']
values = [15, 30, 45, 22]

# Create a bar plot
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(categories, values)

# Add data labels on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height}', ha='center', va='bottom')

# Add labels and title
ax.set_xlabel('Categories')
ax.set_ylabel('Values')
ax.set_title('Bar Plot Example')

plt.show()
```

### Histogram

```python
# Create random data
np.random.seed(42)
data = np.random.normal(0, 1, 1000)  # 1000 points from a standard normal distribution

# Create a histogram
fig, ax = plt.subplots(figsize=(10, 6))
n, bins, patches = ax.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')

# Add a line showing the expected distribution
x = np.linspace(-4, 4, 100)
y = 1/(1 * np.sqrt(2 * np.pi)) * np.exp( - (x - 0)**2 / (2 * 1**2)) * len(data) * (bins[1] - bins[0])
ax.plot(x, y, 'r--', linewidth=2)

# Add labels and title
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.set_title('Histogram Example: Normal Distribution')

plt.show()
```

## Working with Subplots

Subplots allow you to create multiple plots in a single figure.

### Creating Basic Subplots

```python
# Create a figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot different things in each subplot
x = np.linspace(0, 10, 100)

# Top-left: Line plot
axs[0, 0].plot(x, np.sin(x))
axs[0, 0].set_title('Sine Wave')

# Top-right: Scatter plot
axs[0, 1].scatter(x, np.random.rand(100))
axs[0, 1].set_title('Scatter Plot')

# Bottom-left: Bar plot
axs[1, 0].bar(['A', 'B', 'C', 'D'], [10, 20, 15, 25])
axs[1, 0].set_title('Bar Plot')

# Bottom-right: Histogram
axs[1, 1].hist(np.random.normal(0, 1, 1000), bins=30)
axs[1, 1].set_title('Histogram')

# Adjust layout
plt.tight_layout()
plt.show()
```

### Creating Subplots with Different Sizes

```python
# Create a figure with a 2x2 grid
fig = plt.figure(figsize=(12, 10))

# Add subplots with different sizes
ax1 = fig.add_subplot(2, 2, 1)  # Top-left
ax2 = fig.add_subplot(2, 2, 2)  # Top-right
ax3 = fig.add_subplot(2, 1, 2)  # Bottom (spans the width)

# Plot different things in each subplot
ax1.plot(x, np.sin(x))
ax1.set_title('Sine Wave')

ax2.scatter(x, np.random.rand(100))
ax2.set_title('Scatter Plot')

ax3.bar(['A', 'B', 'C', 'D', 'E'], [10, 20, 15, 25, 30])
ax3.set_title('Bar Plot')

# Adjust layout
plt.tight_layout()
plt.show()
```

### Using GridSpec for Complex Layouts

```python
import matplotlib.gridspec as gridspec

# Create a figure
fig = plt.figure(figsize=(12, 10))

# Create a 3x3 grid
gs = gridspec.GridSpec(3, 3)

# Add subplots with different sizes
ax1 = fig.add_subplot(gs[0, :])  # Top row, all columns
ax2 = fig.add_subplot(gs[1, 0:2])  # Middle row, first two columns
ax3 = fig.add_subplot(gs[1:, 2])  # Middle and bottom rows, last column
ax4 = fig.add_subplot(gs[2, 0])  # Bottom row, first column
ax5 = fig.add_subplot(gs[2, 1])  # Bottom row, second column

# Plot different things in each subplot
ax1.plot(x, np.sin(x))
ax1.set_title('Sine Wave')

ax2.scatter(x, np.random.rand(100))
ax2.set_title('Scatter Plot')

ax3.barh(['A', 'B', 'C', 'D', 'E'], [10, 20, 15, 25, 30])
ax3.set_title('Horizontal Bar Plot')

ax4.hist(np.random.normal(0, 1, 1000), bins=30)
ax4.set_title('Histogram')

ax5.plot(x, np.cos(x))
ax5.set_title('Cosine Wave')

# Adjust layout
plt.tight_layout()
plt.show()
```

## Controlling Plot Appearance

Matplotlib provides many options to customize the appearance of your plots.

### Colors, Markers, and Line Styles

```python
# Create data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.cos(x)

# Create a figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot with different colors, markers, and line styles
ax.plot(x, y1, color='blue', linestyle='-', marker='o', markersize=4, label='Sine')
ax.plot(x, y2, color='red', linestyle='--', marker='s', markersize=4, label='Cosine')
ax.plot(x, y3, color='green', linestyle='-.', marker='^', markersize=4, label='Sine*Cosine')

# Add labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Line Styles and Markers Example')
ax.legend()

plt.show()
```

### Color Maps

```python
# Create a 2D array
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# Create a figure and axes
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Plot with different colormaps
im1 = axs[0].imshow(Z, cmap='viridis', origin='lower', extent=[-3, 3, -3, 3])
axs[0].set_title('Viridis Colormap')
fig.colorbar(im1, ax=axs[0])

im2 = axs[1].imshow(Z, cmap='plasma', origin='lower', extent=[-3, 3, -3, 3])
axs[1].set_title('Plasma Colormap')
fig.colorbar(im2, ax=axs[1])

plt.tight_layout()
plt.show()
```

### Setting Axis Limits and Ticks

```python
# Create data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data
ax.plot(x, y)

# Set axis limits
ax.set_xlim(0, 8)
ax.set_ylim(-1.2, 1.2)

# Set custom ticks
ax.set_xticks([0, np.pi, 2*np.pi, 3*np.pi])
ax.set_xticklabels(['0', '$\pi$', '$2\pi$', '$3\pi$'])

# Add a grid
ax.grid(True, linestyle='--', alpha=0.7)

# Add labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Custom Axis Limits and Ticks')

plt.show()
```

## Adding Annotations, Legends, and Labels

Annotations help explain your data and make your plots more informative.

### Adding Text and Arrows

```python
# Create data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data
ax.plot(x, y)

# Add text at a specific point
ax.text(4, 0.8, 'Local Maximum', fontsize=12)

# Add an arrow pointing to a specific feature
ax.annotate('Local Minimum', 
            xy=(7.85, -1),  # Point to annotate
            xytext=(9, -0.5),  # Position of the text
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

# Add labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Text and Arrow Annotations')

plt.show()
```

### Creating Legends

```python
# Create data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.cos(x)

# Create a figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot with labels
line1, = ax.plot(x, y1, label='Sine')
line2, = ax.plot(x, y2, label='Cosine')
line3, = ax.plot(x, y3, label='Sine*Cosine')

# Add a legend with custom position
ax.legend(loc='upper right')

# Alternative: Create a legend with custom properties
# ax.legend([line1, line2, line3], ['Sine', 'Cosine', 'Sine*Cosine'],
#           loc='best', frameon=True, framealpha=0.7, fontsize=10)

# Add labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Legend Example')

plt.show()
```

### Adding Labels and Titles

```python
# Create data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data
ax.plot(x, y)

# Add labels with custom properties
ax.set_xlabel('X-axis', fontsize=14, fontweight='bold')
ax.set_ylabel('Y-axis', fontsize=14, fontweight='bold')

# Add a title and subtitle
ax.set_title('Main Title', fontsize=16, fontweight='bold')
ax.text(0.5, 1.05, 'Subtitle or Additional Information', 
        transform=ax.transAxes, ha='center', fontsize=12)

# Add a text box with information
textstr = 'Some stats:\n$\mu=%.2f$\n$\sigma=%.2f$' % (np.mean(y), np.std(y))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.show()
```

## Practice Exercises

1. Create a line plot showing multiple sine waves with different frequencies.
2. Generate a scatter plot with points colored according to a third variable.
3. Create a bar chart comparing sales data across different categories.
4. Make a histogram of random data and overlay a probability density function.
5. Create a figure with 4 subplots showing different visualizations of the same dataset.
6. Customize a plot with specific colors, markers, and line styles.
7. Add annotations to highlight important features in your plot.
8. Create a plot with a custom legend and meaningful labels.

## Key Takeaways

- Matplotlib provides a flexible foundation for creating a wide variety of visualizations
- Understanding the Figure and Axes objects is key to creating effective plots
- The object-oriented interface offers more control for complex visualizations
- Basic plot types include line plots, scatter plots, bar plots, and histograms
- Subplots allow you to combine multiple visualizations in a single figure
- Customizing appearance through colors, markers, and styles helps convey information effectively
- Annotations, legends, and labels make your plots more informative and easier to understand
