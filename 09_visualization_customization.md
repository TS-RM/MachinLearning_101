# Visualization Customization Techniques

## Introduction to Visualization Customization

Creating effective data visualizations goes beyond just plotting data. Customization techniques help you enhance the clarity, aesthetics, and impact of your visualizations. This guide covers various customization techniques including color palettes, figure sizing, axes customization, and creating publication-quality visualizations.

## Color Palettes and Colormaps

Colors play a crucial role in data visualization. They can highlight patterns, distinguish categories, and represent values.

### Understanding Color Palettes

Color palettes are collections of colors used in visualizations. They can be:

- **Sequential**: Show progression from low to high values (e.g., light blue to dark blue)
- **Diverging**: Highlight deviations from a central value (e.g., blue to white to red)
- **Qualitative**: Distinguish between categorical data (e.g., distinct colors for different categories)

### Using Matplotlib Colormaps

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# Create sample data
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# Display different colormaps
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm']

for i, cmap_name in enumerate(colormaps):
    im = axes[i].imshow(Z, cmap=cmap_name, origin='lower', extent=[0, 10, 0, 10])
    axes[i].set_title(cmap_name)
    fig.colorbar(im, ax=axes[i])

plt.tight_layout()
plt.show()
```

### Creating Custom Color Palettes with Seaborn

```python
# Display Seaborn color palettes
plt.figure(figsize=(15, 10))

# Qualitative palettes
plt.subplot(3, 1, 1)
sns.palplot(sns.color_palette("Set1", 9))
plt.title("Qualitative Palette: Set1")

# Sequential palettes
plt.subplot(3, 1, 2)
sns.palplot(sns.color_palette("Blues", 9))
plt.title("Sequential Palette: Blues")

# Diverging palettes
plt.subplot(3, 1, 3)
sns.palplot(sns.color_palette("RdBu", 9))
plt.title("Diverging Palette: RdBu")

plt.tight_layout()
plt.show()

# Create a custom color palette
custom_palette = sns.color_palette("husl", 8)
plt.figure(figsize=(10, 2))
sns.palplot(custom_palette)
plt.title("Custom Palette using HUSL color space")
plt.show()
```

### Choosing the Right Colormap

- **Sequential colormaps** (e.g., 'viridis', 'Blues'): Best for continuous data where higher values are more important
- **Diverging colormaps** (e.g., 'coolwarm', 'RdBu'): Best for data with a meaningful midpoint (like zero)
- **Qualitative colormaps** (e.g., 'Set1', 'tab10'): Best for categorical data
- **Perceptually uniform colormaps** (e.g., 'viridis', 'plasma'): Maintain perceptual differences across the range

### Applying Colormaps to Different Plot Types

```python
# Scatter plot with colormap
plt.figure(figsize=(10, 6))
scatter = plt.scatter(np.random.rand(100), np.random.rand(100), 
                     c=np.random.rand(100), cmap='viridis', 
                     s=100, alpha=0.7)
plt.colorbar(scatter, label='Value')
plt.title('Scatter Plot with Colormap')
plt.tight_layout()
plt.show()

# Line plot with colormap
plt.figure(figsize=(10, 6))
for i in range(10):
    plt.plot(np.arange(10), np.arange(10) + i, 
             color=plt.cm.viridis(i/10), 
             linewidth=2, 
             label=f'Line {i+1}')
plt.title('Line Plot with Colormap')
plt.legend()
plt.tight_layout()
plt.show()

# Bar chart with colormap
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
values = np.random.randint(10, 100, len(categories))

plt.figure(figsize=(10, 6))
bars = plt.bar(categories, values, color=plt.cm.plasma(np.linspace(0, 1, len(categories))))
plt.title('Bar Chart with Colormap')
plt.tight_layout()
plt.show()
```

## Controlling Figure Size and Resolution

The size and resolution of your visualizations affect their clarity and impact, especially when sharing or publishing them.

### Setting Figure Size

```python
# Basic figure size setting
plt.figure(figsize=(10, 6))  # Width: 10 inches, Height: 6 inches
plt.plot(np.random.randn(100).cumsum())
plt.title('Basic Plot with Custom Figure Size')
plt.tight_layout()
plt.show()

# Figure size for subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, ax in enumerate(axes.flatten()):
    ax.plot(np.random.randn(100).cumsum())
    ax.set_title(f'Subplot {i+1}')
plt.tight_layout()
plt.show()
```

### Controlling DPI (Dots Per Inch)

```python
# Set DPI for display
plt.figure(figsize=(8, 6), dpi=100)  # Default is usually 100 DPI
plt.plot(np.random.randn(100).cumsum())
plt.title('Plot with Specified DPI')
plt.tight_layout()
plt.show()

# Save figure with high DPI for publication
plt.figure(figsize=(8, 6))
plt.plot(np.random.randn(100).cumsum())
plt.title('High Resolution Plot for Publication')
plt.tight_layout()
plt.savefig('high_res_plot.png', dpi=300)  # 300 DPI for publication quality
plt.close()
```

### Adjusting Layout

```python
# Using tight_layout
plt.figure(figsize=(10, 6))
plt.plot(np.random.randn(100).cumsum())
plt.title('Plot with Tight Layout')
plt.xlabel('X-axis with a very long label that might get cut off')
plt.ylabel('Y-axis with a very long label that might get cut off')
plt.tight_layout()  # Adjusts padding to fit all elements
plt.show()

# Manual layout adjustment
plt.figure(figsize=(10, 6))
plt.plot(np.random.randn(100).cumsum())
plt.title('Plot with Manual Layout Adjustment')
plt.xlabel('X-axis with a very long label')
plt.ylabel('Y-axis with a very long label')
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
plt.show()
```

### Saving Figures in Different Formats

```python
# Create a sample plot
plt.figure(figsize=(10, 6))
plt.plot(np.random.randn(100).cumsum())
plt.title('Sample Plot for Saving')

# Save in different formats
plt.savefig('plot.png')  # PNG format
plt.savefig('plot.jpg', quality=95)  # JPEG format with quality setting
plt.savefig('plot.svg')  # SVG format for vector graphics
plt.savefig('plot.pdf')  # PDF format
plt.close()
```

## Customizing Axes, Ticks, and Gridlines

Customizing axes helps improve the readability and appearance of your visualizations.

### Customizing Axes Limits and Scales

```python
# Set axis limits
plt.figure(figsize=(10, 6))
plt.plot(np.random.randn(100).cumsum())
plt.xlim(0, 80)  # Set x-axis limits
plt.ylim(-10, 10)  # Set y-axis limits
plt.title('Plot with Custom Axis Limits')
plt.tight_layout()
plt.show()

# Log scale
plt.figure(figsize=(10, 6))
plt.plot(np.exp(np.linspace(0, 5, 100)))
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.title('Plot with Logarithmic Y-axis')
plt.tight_layout()
plt.show()

# Symlog scale (linear near zero, logarithmic away from zero)
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(-100, 100, 100), np.linspace(-100, 100, 100)**3)
plt.yscale('symlog')  # Symmetric log scale
plt.title('Plot with Symmetric Log Y-axis')
plt.tight_layout()
plt.show()
```

### Customizing Ticks

```python
# Custom tick positions and labels
plt.figure(figsize=(10, 6))
plt.plot(np.sin(np.linspace(0, 2*np.pi, 100)))
plt.xticks([0, 25, 50, 75, 99], ['0', 'π/2', 'π', '3π/2', '2π'])
plt.yticks([-1, -0.5, 0, 0.5, 1], ['Min', '', 'Zero', '', 'Max'])
plt.title('Plot with Custom Tick Labels')
plt.tight_layout()
plt.show()

# Rotating tick labels
plt.figure(figsize=(10, 6))
plt.bar(range(12), np.random.randint(1, 100, 12))
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 
          rotation=45)
plt.title('Bar Chart with Rotated Tick Labels')
plt.tight_layout()
plt.show()

# Formatting tick labels
import matplotlib.ticker as ticker

plt.figure(figsize=(10, 6))
plt.plot(np.random.randint(1000, 5000, 20))
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
plt.title('Plot with Formatted Y-axis Tick Labels')
plt.tight_layout()
plt.show()
```

### Customizing Gridlines

```python
# Basic grid
plt.figure(figsize=(10, 6))
plt.plot(np.random.randn(100).cumsum())
plt.grid(True)
plt.title('Plot with Basic Grid')
plt.tight_layout()
plt.show()

# Customized grid
plt.figure(figsize=(10, 6))
plt.plot(np.random.randn(100).cumsum())
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.title('Plot with Customized Grid')
plt.tight_layout()
plt.show()

# Different grids for major and minor ticks
plt.figure(figsize=(10, 6))
plt.plot(np.random.randn(100).cumsum())
plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)
plt.minorticks_on()
plt.title('Plot with Major and Minor Gridlines')
plt.tight_layout()
plt.show()
```

## Adding Titles, Subtitles, and Captions

Titles, subtitles, and captions help provide context and explanation for your visualizations.

### Adding Main Title and Axis Labels

```python
plt.figure(figsize=(10, 6))
plt.plot(np.random.randn(100).cumsum())

# Add title and axis labels
plt.title('Main Title', fontsize=16, fontweight='bold')
plt.xlabel('X-axis Label', fontsize=12)
plt.ylabel('Y-axis Label', fontsize=12)

plt.tight_layout()
plt.show()
```

### Adding Subtitles

```python
plt.figure(figsize=(10, 6))
plt.plot(np.random.randn(100).cumsum())

# Add title
plt.title('Main Title', fontsize=16, fontweight='bold')

# Add subtitle
plt.suptitle('Subtitle or Additional Information', fontsize=12, y=0.92)

# Add axis labels
plt.xlabel('X-axis Label', fontsize=12)
plt.ylabel('Y-axis Label', fontsize=12)

plt.tight_layout()
plt.subplots_adjust(top=0.85)  # Make room for suptitle
plt.show()
```

### Adding Captions and Annotations

```python
plt.figure(figsize=(10, 6))
plt.plot(np.random.randn(100).cumsum())

# Add title
plt.title('Main Title', fontsize=16, fontweight='bold')

# Add axis labels
plt.xlabel('X-axis Label', fontsize=12)
plt.ylabel('Y-axis Label', fontsize=12)

# Add caption at the bottom
plt.figtext(0.5, 0.01, 'Caption: Additional information about the figure.', 
           ha='center', fontsize=10, style='italic')

# Add annotation to highlight a specific point
plt.annotate('Important Point', xy=(50, 5), xytext=(60, 10),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Make room for caption
plt.show()
```

### Using Text Boxes

```python
plt.figure(figsize=(10, 6))
plt.plot(np.random.randn(100).cumsum())

# Add title
plt.title('Main Title', fontsize=16, fontweight='bold')

# Add a text box with statistics
textstr = 'Some statistics:\n$\mu=%.2f$\n$\sigma=%.2f$' % (0, 1)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()
```

## Creating Publication-Quality Visualizations

Publication-quality visualizations meet high standards for clarity, aesthetics, and professionalism.

### Setting a Professional Style

```python
# Use a professional style
plt.style.use('seaborn-v0_8-whitegrid')  # Clean, professional style

plt.figure(figsize=(10, 6))
plt.plot(np.random.randn(100).cumsum())
plt.title('Plot with Professional Style', fontsize=14)
plt.xlabel('X-axis', fontsize=12)
plt.ylabel('Y-axis', fontsize=12)
plt.tight_layout()
plt.show()
```

### Creating a Custom Style

```python
# Define custom style parameters
custom_style = {
    'figure.figsize': (10, 6),
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7
}

# Apply custom style
with plt.style.context(custom_style):
    plt.figure()
    plt.plot(np.random.randn(100).cumsum())
    plt.title('Plot with Custom Style')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.tight_layout()
    plt.show()
```

### Multi-Panel Figures for Publications

```python
# Create a multi-panel figure
fig = plt.figure(figsize=(12, 10))

# Add panel labels
labels = ['A', 'B', 'C', 'D']
gs = fig.add_gridspec(2, 2)

for i, label in enumerate(labels):
    row, col = i // 2, i % 2
    ax = fig.add_subplot(gs[row, col])
    
    # Add data to each panel
    if i == 0:
        ax.plot(np.random.randn(100).cumsum())
        ax.set_title('Time Series')
    elif i == 1:
        ax.scatter(np.random.randn(100), np.random.randn(100))
        ax.set_title('Scatter Plot')
    elif i == 2:
        ax.bar(range(10), np.random.randint(1, 10, 10))
        ax.set_title('Bar Chart')
    else:
        ax.hist(np.random.randn(1000), bins=30)
        ax.set_title('Histogram')
    
    # Add panel label
    ax.text(-0.1, 1.1, label, transform=ax.transAxes, fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()
```

### Consistent Color and Style Across Multiple Figures

```python
# Set a consistent style
plt.style.use('seaborn-v0_8-whitegrid')

# Set a consistent color palette
colors = plt.cm.viridis(np.linspace(0, 1, 4))

# Create multiple figures with consistent style
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(np.random.randn(100).cumsum(), color=colors[0])
ax1.set_title('Figure 1: Time Series')

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.scatter(np.random.randn(100), np.random.randn(100), color=colors[1])
ax2.set_title('Figure 2: Scatter Plot')

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.bar(range(10), np.random.randint(1, 10, 10), color=colors[2])
ax3.set_title('Figure 3: Bar Chart')

fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.hist(np.random.randn(1000), bins=30, color=colors[3])
ax4.set_title('Figure 4: Histogram')

plt.show()
```

### Exporting High-Resolution Figures for Publication

```python
# Create a publication-quality figure
plt.figure(figsize=(8, 6))
plt.plot(np.random.randn(100).cumsum(), linewidth=2)
plt.title('Publication-Quality Figure', fontsize=14, fontweight='bold')
plt.xlabel('X-axis', fontsize=12)
plt.ylabel('Y-axis', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save in high resolution for publication
plt.savefig('publication_figure.png', dpi=300, bbox_inches='tight')
plt.savefig('publication_figure.pdf', bbox_inches='tight')  # Vector format for highest quality
plt.savefig('publication_figure.svg', bbox_inches='tight')  # SVG for web
plt.close()
```

## Practice Exercises

1. Create a visualization using a custom color palette that effectively represents your data.
2. Design a multi-panel figure with consistent styling across all panels.
3. Customize the axes, ticks, and gridlines of a plot to improve its readability.
4. Create a publication-quality figure with a main title, subtitle, and caption.
5. Export a high-resolution figure in multiple formats suitable for different purposes.
6. Create a custom style and apply it to multiple visualizations for consistency.
7. Design a visualization with custom annotations to highlight important features.
8. Create a figure with both major and minor gridlines and custom tick formatting.

## Key Takeaways

- Color palettes and colormaps should be chosen based on the type of data and the message you want to convey
- Figure size and resolution affect the clarity and impact of your visualizations
- Customizing axes, ticks, and gridlines improves readability and aesthetics
- Titles, subtitles, and captions provide context and explanation for your visualizations
- Publication-quality visualizations require attention to detail in style, consistency, and resolution
- A consistent style across multiple figures helps create a cohesive narrative
- High-resolution exports in appropriate formats ensure your visualizations look professional in any medium
