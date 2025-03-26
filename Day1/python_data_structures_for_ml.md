# Python Data Structures Refresher for Machine Learning

This guide provides a comprehensive refresher on Python data structures with a focus on applications relevant to machine learning. Each section includes explanations, examples, and practical applications.

## Table of Contents
1. [Numbers and Basic Operations](#numbers-and-basic-operations)
2. [Lists](#lists)
3. [Tuples](#tuples)
4. [Dictionaries](#dictionaries)
5. [Sets](#sets)
6. [Strings and Text Processing](#strings-and-text-processing)
7. [NumPy Arrays](#numpy-arrays)
8. [Pandas DataFrames](#pandas-dataframes)
9. [Custom Data Structures](#custom-data-structures)
10. [ML-Specific Applications](#ml-specific-applications)

## Numbers and Basic Operations

### Basic Types
```python
# Integers
x = 5
y = -10

# Floating point numbers
a = 3.14
b = -0.001

# Complex numbers (useful in some ML algorithms)
c = 2 + 3j
```

### Arithmetic Operations
```python
# Basic operations
addition = 5 + 3        # 8
subtraction = 5 - 3     # 2
multiplication = 5 * 3  # 15
division = 5 / 3        # 1.6666...
floor_division = 5 // 3 # 1
modulo = 5 % 3          # 2
exponentiation = 5 ** 3 # 125

# Common math functions (import math for more)
import math
sqrt_value = math.sqrt(16)  # 4.0
log_value = math.log(100)   # 4.605...
sin_value = math.sin(math.pi/2)  # 1.0
```

### ML Application
In machine learning, numerical operations form the foundation of algorithms:
- Feature scaling (normalization, standardization)
- Distance calculations (Euclidean, Manhattan)
- Error calculations (mean squared error, log loss)

```python
# Example: Feature standardization (z-score normalization)
values = [2, 4, 6, 8, 10]
mean = sum(values) / len(values)
std_dev = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
standardized = [(x - mean) / std_dev for x in values]
print(standardized)  # [-1.41, -0.71, 0, 0.71, 1.41]
```

## Lists

Lists are ordered, mutable collections that can store elements of different types.

### Creating Lists
```python
# Empty list
empty_list = []
empty_list_alt = list()

# List with elements
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]

# List comprehension
squares = [x**2 for x in range(10)]  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

### Accessing Elements
```python
numbers = [10, 20, 30, 40, 50]

# Indexing (0-based)
first = numbers[0]      # 10
last = numbers[-1]      # 50

# Slicing [start:end:step]
subset = numbers[1:4]   # [20, 30, 40]
reversed_list = numbers[::-1]  # [50, 40, 30, 20, 10]
```

### Common Operations
```python
numbers = [1, 2, 3]

# Adding elements
numbers.append(4)       # [1, 2, 3, 4]
numbers.insert(1, 1.5)  # [1, 1.5, 2, 3, 4]
numbers.extend([5, 6])  # [1, 1.5, 2, 3, 4, 5, 6]

# Removing elements
numbers.remove(1.5)     # [1, 2, 3, 4, 5, 6]
popped = numbers.pop()  # popped = 6, numbers = [1, 2, 3, 4, 5]
popped_index = numbers.pop(1)  # popped_index = 2, numbers = [1, 3, 4, 5]

# Finding elements
index = numbers.index(3)  # 1
count = numbers.count(1)  # 1

# Sorting
numbers.sort()          # In-place sorting
numbers.sort(reverse=True)  # Descending order
sorted_numbers = sorted(numbers)  # Returns a new sorted list
```

### ML Application
Lists are commonly used in ML for:
- Storing sequences of data points
- Collecting model predictions
- Managing hyperparameters

```python
# Example: K-fold cross-validation indices
def k_fold_split(data, k=5):
    fold_size = len(data) // k
    indices = list(range(len(data)))
    folds = []
    for i in range(k):
        test_indices = indices[i*fold_size:(i+1)*fold_size]
        train_indices = [idx for idx in indices if idx not in test_indices]
        folds.append((train_indices, test_indices))
    return folds
```

## Tuples

Tuples are ordered, immutable collections that can store elements of different types.

### Creating Tuples
```python
# Empty tuple
empty_tuple = ()
empty_tuple_alt = tuple()

# Tuple with elements
coordinates = (10, 20)
mixed_tuple = (1, "hello", 3.14)

# Single element tuple (note the comma)
single = (42,)
```

### Accessing Elements
```python
coordinates = (10, 20, 30, 40, 50)

# Indexing (0-based)
x = coordinates[0]      # 10
z = coordinates[2]      # 30

# Slicing [start:end:step]
subset = coordinates[1:4]  # (20, 30, 40)
```

### Common Operations
```python
# Concatenation
tuple1 = (1, 2, 3)
tuple2 = (4, 5, 6)
combined = tuple1 + tuple2  # (1, 2, 3, 4, 5, 6)

# Repetition
repeated = tuple1 * 3  # (1, 2, 3, 1, 2, 3, 1, 2, 3)

# Unpacking
x, y, z = (10, 20, 30)  # x=10, y=20, z=30

# Finding elements
coordinates = (10, 20, 30, 20, 40)
index = coordinates.index(20)  # 1 (first occurrence)
count = coordinates.count(20)  # 2
```

### ML Application
Tuples are useful in ML for:
- Representing fixed feature sets
- Storing model hyperparameters
- Returning multiple values from functions

```python
# Example: Grid search hyperparameters
hyperparameter_grid = [
    ('learning_rate', [0.001, 0.01, 0.1]),
    ('batch_size', [32, 64, 128]),
    ('activation', ['relu', 'tanh', 'sigmoid'])
]

# Example: Confusion matrix coordinates
def get_confusion_matrix_elements(y_true, y_pred):
    tp = sum((a == 1 and b == 1) for a, b in zip(y_true, y_pred))
    fp = sum((a == 0 and b == 1) for a, b in zip(y_true, y_pred))
    fn = sum((a == 1 and b == 0) for a, b in zip(y_true, y_pred))
    tn = sum((a == 0 and b == 0) for a, b in zip(y_true, y_pred))
    return (tp, fp, fn, tn)
```

## Dictionaries

Dictionaries are unordered, mutable collections of key-value pairs.

### Creating Dictionaries
```python
# Empty dictionary
empty_dict = {}
empty_dict_alt = dict()

# Dictionary with elements
person = {'name': 'John', 'age': 30, 'city': 'New York'}

# Using dict() constructor
person_alt = dict(name='John', age=30, city='New York')

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

### Accessing Elements
```python
person = {'name': 'John', 'age': 30, 'city': 'New York'}

# Using keys
name = person['name']  # 'John'

# Using get() (safer, returns None or default if key doesn't exist)
age = person.get('age')  # 30
country = person.get('country', 'Unknown')  # 'Unknown'
```

### Common Operations
```python
person = {'name': 'John', 'age': 30}

# Adding/updating elements
person['city'] = 'New York'  # {'name': 'John', 'age': 30, 'city': 'New York'}
person.update({'age': 31, 'job': 'Engineer'})  # {'name': 'John', 'age': 31, 'city': 'New York', 'job': 'Engineer'}

# Removing elements
city = person.pop('city')  # city = 'New York', person = {'name': 'John', 'age': 31, 'job': 'Engineer'}
job, value = person.popitem()  # Removes last inserted item
person.clear()  # Empties the dictionary

# Checking keys
person = {'name': 'John', 'age': 30}
'name' in person  # True
'city' in person  # False

# Getting views
keys = person.keys()  # dict_keys(['name', 'age'])
values = person.values()  # dict_values(['John', 30])
items = person.items()  # dict_items([('name', 'John'), ('age', 30)])
```

### ML Application
Dictionaries are extensively used in ML for:
- Storing model parameters
- Managing feature mappings
- Configuring model settings

```python
# Example: One-hot encoding categorical features
def one_hot_encode(categories):
    unique_categories = list(set(categories))
    encoding_dict = {category: i for i, category in enumerate(unique_categories)}
    
    encoded = []
    for category in categories:
        one_hot = [0] * len(unique_categories)
        one_hot[encoding_dict[category]] = 1
        encoded.append(one_hot)
    
    return encoded, encoding_dict

# Example: Model configuration
model_config = {
    'architecture': 'CNN',
    'layers': [
        {'type': 'conv2d', 'filters': 32, 'kernel_size': 3, 'activation': 'relu'},
        {'type': 'maxpool2d', 'pool_size': 2},
        {'type': 'flatten'},
        {'type': 'dense', 'units': 128, 'activation': 'relu'},
        {'type': 'dense', 'units': 10, 'activation': 'softmax'}
    ],
    'optimizer': 'adam',
    'learning_rate': 0.001
}
```

## Sets

Sets are unordered, mutable collections of unique elements.

### Creating Sets
```python
# Empty set (note: {} creates an empty dictionary)
empty_set = set()

# Set with elements
numbers = {1, 2, 3, 4, 5}
mixed_set = {1, 'hello', 3.14}

# Set from iterable
list_to_set = set([1, 2, 2, 3, 3, 3])  # {1, 2, 3}

# Set comprehension
even_squares = {x**2 for x in range(10) if x % 2 == 0}  # {0, 4, 16, 36, 64}
```

### Common Operations
```python
# Adding elements
numbers = {1, 2, 3}
numbers.add(4)  # {1, 2, 3, 4}
numbers.update([4, 5, 6])  # {1, 2, 3, 4, 5, 6}

# Removing elements
numbers.remove(6)  # Raises KeyError if element doesn't exist
numbers.discard(7)  # No error if element doesn't exist
popped = numbers.pop()  # Removes and returns an arbitrary element
numbers.clear()  # Empties the set

# Set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

union = set1 | set2  # or set1.union(set2)  # {1, 2, 3, 4, 5, 6}
intersection = set1 & set2  # or set1.intersection(set2)  # {3, 4}
difference = set1 - set2  # or set1.difference(set2)  # {1, 2}
symmetric_diff = set1 ^ set2  # or set1.symmetric_difference(set2)  # {1, 2, 5, 6}

# Checking subsets/supersets
{1, 2}.issubset({1, 2, 3})  # True
{1, 2, 3}.issuperset({1, 2})  # True
```

### ML Application
Sets are useful in ML for:
- Feature selection (unique features)
- Removing duplicates in datasets
- Tracking unique classes in classification

```python
# Example: Finding unique features in text data
def extract_unique_words(documents):
    all_words = set()
    for doc in documents:
        words = doc.lower().split()
        all_words.update(words)
    return all_words

# Example: Feature intersection
def common_features(feature_set1, feature_set2):
    return feature_set1.intersection(feature_set2)
```

## Strings and Text Processing

Strings are immutable sequences of characters, crucial for text processing in ML.

### String Basics
```python
# Creating strings
single_quotes = 'Hello'
double_quotes = "World"
triple_quotes = '''Multiline
string'''

# String operations
greeting = "Hello" + " " + "World"  # Concatenation
repeated = "Python " * 3  # "Python Python Python "

# String methods
text = "  Python for Machine Learning  "
text.upper()  # "  PYTHON FOR MACHINE LEARNING  "
text.lower()  # "  python for machine learning  "
text.strip()  # "Python for Machine Learning"
text.replace("Python", "Coding")  # "  Coding for Machine Learning  "
text.split()  # ["Python", "for", "Machine", "Learning"]
"-".join(["Python", "ML"])  # "Python-ML"

# String formatting
name = "Alice"
age = 30
f"Name: {name}, Age: {age}"  # "Name: Alice, Age: 30"
"Name: {}, Age: {}".format(name, age)  # "Name: Alice, Age: 30"
```

### String Indexing and Slicing
```python
text = "Python"
text[0]  # 'P'
text[-1]  # 'n'
text[0:3]  # 'Pyt'
text[::2]  # 'Pto'
```

### ML Application
String processing is essential for NLP tasks:
- Text preprocessing
- Feature extraction
- Tokenization

```python
# Example: Basic text preprocessing for ML
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    import string
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Tokenize
    tokens = text.split()
    
    # Remove stop words
    stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'and'}
    tokens = [token for token in tokens if token not in stop_words]
    
    return tokens

# Example: Bag of words representation
def create_bow(documents):
    word_set = set()
    for doc in documents:
        tokens = preprocess_text(doc)
        word_set.update(tokens)
    
    word_to_index = {word: i for i, word in enumerate(sorted(word_set))}
    
    bow_vectors = []
    for doc in documents:
        tokens = preprocess_text(doc)
        vector = [0] * len(word_set)
        for token in tokens:
            vector[word_to_index[token]] += 1
        bow_vectors.append(vector)
    
    return bow_vectors, word_to_index
```

## NumPy Arrays

NumPy arrays are the foundation of numerical computing in Python and essential for ML.

### Creating NumPy Arrays
```python
import numpy as np

# From Python lists
arr1d = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])

# Special arrays
zeros = np.zeros((3, 3))  # 3x3 array of zeros
ones = np.ones((2, 4))  # 2x4 array of ones
identity = np.eye(3)  # 3x3 identity matrix
random_uniform = np.random.rand(2, 2)  # 2x2 array of random values [0,1)
random_normal = np.random.randn(2, 2)  # 2x2 array from standard normal distribution
range_array = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)  # 5 evenly spaced points between 0 and 1
```

### Array Indexing and Slicing
```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Indexing
arr[0, 0]  # 1
arr[2, 1]  # 8

# Slicing
arr[0:2, 1:3]  # array([[2, 3], [5, 6]])
arr[:, 1]  # array([2, 5, 8])  # Second column

# Boolean indexing
arr[arr > 5]  # array([6, 7, 8, 9])
```

### Array Operations
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise operations
a + b  # array([5, 7, 9])
a * b  # array([4, 10, 18])
a ** 2  # array([1, 4, 9])

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

np.dot(A, B)  # array([[19, 22], [43, 50]])
A.dot(B)  # Same as above

# Aggregation
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr.sum()  # 21
arr.sum(axis=0)  # array([5, 7, 9])  # Sum of each column
arr.sum(axis=1)  # array([6, 15])  # Sum of each row
arr.mean()  # 3.5
arr.std()  # Standard deviation
arr.min()  # 1
arr.max()  # 6
```

### Array Reshaping and Manipulation
```python
arr = np.arange(12)

# Reshaping
arr.reshape(3, 4)  # 3x4 array
arr.reshape(2, 2, 3)  # 2x2x3 array

# Transposing
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_2d.T  # Transpose

# Stacking
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.vstack((a, b))  # Vertical stack: array([[1, 2, 3], [4, 5, 6]])
np.hstack((a, b))  # Horizontal stack: array([1, 2, 3, 4, 5, 6])
```

### ML Application
NumPy arrays are fundamental in ML for:
- Representing feature matrices and target vectors
- Implementing mathematical operations efficiently
- Performing linear algebra operations

```python
# Example: Linear regression using NumPy
def linear_regression(X, y, learning_rate=0.01, iterations=1000):
    # Add bias term
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # Initialize weights
    theta = np.random.randn(X_b.shape[1])
    
    # Gradient descent
    for i in range(iterations):
        gradients = 2/X_b.shape[0] * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - learning_rate * gradients
    
    return theta

# Example: Feature scaling
def standardize_features(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std
```

## Pandas DataFrames

Pandas DataFrames are essential for data manipulation and analysis in ML.

### Creating DataFrames
```python
import pandas as pd

# From dictionaries
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Paris', 'London']
}
df = pd.DataFrame(data)

# From lists
data_list = [
    ['Alice', 25, 'New York'],
    ['Bob', 30, 'Paris'],
    ['Charlie', 35, 'London']
]
df = pd.DataFrame(data_list, columns=['Name', 'Age', 'City'])

# From CSV
df = pd.read_csv('data.csv')

# From NumPy array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])
```

### Accessing Data
```python
# Basic information
df.head()  # First 5 rows
df.tail()  # Last 5 rows
df.info()  # Summary info
df.describe()  # Statistical summary
df.shape  # (rows, columns)

# Selecting columns
df['Name']  # Single column as Series
df[['Name', 'Age']]  # Multiple columns as DataFrame

# Selecting rows
df.loc[0]  # Row by label
df.iloc[0]  # Row by position
df.loc[0:2, 'Name':'City']  # Rows and columns by label
df.iloc[0:2, 0:2]  # Rows and columns by position

# Boolean indexing
df[df['Age'] > 30]  # Rows where Age > 30
df[(df['Age'] > 25) & (df['City'] == 'London')]  # Combined conditions
```

### Data Manipulation
```python
# Adding columns
df['Country'] = ['USA', 'France', 'UK']

# Applying functions
df['Age_Squared'] = df['Age'].apply(lambda x: x**2)

# Sorting
df.sort_values('Age', ascending=False)  # Sort by Age descending
df.sort_values(['City', 'Age'])  # Sort by multiple columns

# Grouping
df.groupby('City').mean()  # Mean of numeric columns by City
df.groupby('City').agg({'Age': ['min', 'max', 'mean']})  # Multiple aggregations

# Merging
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Alice', 'Bob', 'Charlie']})
df2 = pd.DataFrame({'ID': [1, 2, 4], 'Age': [25, 30, 35]})
pd.merge(df1, df2, on='ID')  # Inner join on ID
pd.merge(df1, df2, on='ID', how='left')  # Left join on ID

# Handling missing values
df.isna().sum()  # Count missing values
df.fillna(0)  # Fill missing values with 0
df.dropna()  # Drop rows with any missing values
```

### ML Application
Pandas DataFrames are crucial in ML for:
- Data preprocessing and cleaning
- Feature engineering
- Exploratory data analysis

```python
# Example: Data preprocessing pipeline
def preprocess_data(df):
    # Handle missing values
    df = df.fillna(df.mean())
    
    # One-hot encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Feature scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    numeric_cols = df_encoded.select_dtypes(include=['float64', 'int64']).columns
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
    
    return df_encoded

# Example: Train-test split
def train_test_split_df(df, target_col, test_size=0.2, random_state=42):
    import numpy as np
    
    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1, random_state=random_state)
    
    # Split into features and target
    X = df_shuffled.drop(target_col, axis=1)
    y = df_shuffled[target_col]
    
    # Calculate split index
    split_idx = int(len(df) * (1 - test_size))
    
    # Split the data
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test
```

## Custom Data Structures

Sometimes, you'll need to create custom data structures for specific ML tasks.

### Classes and Objects
```python
class DataPoint:
    def __init__(self, features, label=None):
        self.features = features
        self.label = label
        self.predicted_label = None
    
    def distance(self, other):
        """Calculate Euclidean distance to another data point"""
        import math
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.features, other.features)))
    
    def __str__(self):
        return f"DataPoint(features={self.features}, label={self.label})"

# Using the custom data structure
point1 = DataPoint([1, 2, 3], "Class A")
point2 = DataPoint([4, 5, 6], "Class B")
print(point1.distance(point2))  # 5.196...
```

### Custom Collections
```python
class Dataset:
    def __init__(self, name):
        self.name = name
        self.data_points = []
        self.feature_names = []
    
    def add_point(self, features, label=None):
        self.data_points.append(DataPoint(features, label))
    
    def set_feature_names(self, names):
        self.feature_names = names
    
    def split(self, test_ratio=0.2):
        """Split dataset into training and test sets"""
        import random
        random.shuffle(self.data_points)
        split_idx = int(len(self.data_points) * (1 - test_ratio))
        
        train_set = Dataset(f"{self.name}_train")
        train_set.feature_names = self.feature_names
        train_set.data_points = self.data_points[:split_idx]
        
        test_set = Dataset(f"{self.name}_test")
        test_set.feature_names = self.feature_names
        test_set.data_points = self.data_points[split_idx:]
        
        return train_set, test_set
```

### ML Application
Custom data structures can be useful for:
- Representing complex ML models
- Managing datasets with specific requirements
- Implementing custom algorithms

```python
# Example: K-Nearest Neighbors implementation with custom data structures
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.training_data = []
    
    def fit(self, dataset):
        self.training_data = dataset.data_points
    
    def predict(self, data_point):
        # Calculate distances to all training points
        distances = [(train_point, train_point.distance(data_point)) 
                     for train_point in self.training_data]
        
        # Sort by distance and take k nearest
        distances.sort(key=lambda x: x[1])
        nearest = distances[:self.k]
        
        # Count labels of nearest neighbors
        from collections import Counter
        votes = Counter(point.label for point, _ in nearest)
        
        # Return most common label
        return votes.most_common(1)[0][0]
    
    def evaluate(self, test_dataset):
        correct = 0
        for point in test_dataset.data_points:
            prediction = self.predict(point)
            if prediction == point.label:
                correct += 1
        
        return correct / len(test_dataset.data_points)
```

## ML-Specific Applications

This section covers how these data structures are specifically applied in machine learning workflows.

### Feature Engineering
```python
# One-hot encoding categorical features
def one_hot_encode(df, categorical_cols):
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Scaling numerical features
def scale_features(X):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# Creating polynomial features
def add_polynomial_features(X, degree=2):
    import numpy as np
    poly_features = []
    for i in range(X.shape[1]):
        for power in range(2, degree + 1):
            poly_features.append(X[:, i] ** power)
    return np.column_stack((X, np.column_stack(poly_features)))
```

### Data Pipelines
```python
# Simple ML pipeline using Python data structures
def ml_pipeline(train_data, test_data, target_col, model_type='random_forest'):
    # Split data
    X_train = train_data.drop(target_col, axis=1)
    y_train = train_data[target_col]
    X_test = test_data.drop(target_col, axis=1)
    y_test = test_data[target_col]
    
    # Preprocess data
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    X_train_encoded = one_hot_encode(X_train, categorical_cols)
    X_test_encoded = one_hot_encode(X_test, categorical_cols)
    
    # Ensure test data has same columns as train data
    missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
    for col in missing_cols:
        X_test_encoded[col] = 0
    X_test_encoded = X_test_encoded[X_train_encoded.columns]
    
    # Scale features
    X_train_scaled = scale_features(X_train_encoded)
    X_test_scaled = scale_features(X_test_encoded)
    
    # Train model
    if model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42)
    else:
        raise ValueError("Unsupported model type")
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    
    return {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'feature_importance': dict(zip(X_train_encoded.columns, model.feature_importances_)) 
                             if hasattr(model, 'feature_importances_') else None
    }
```

### Model Representation
```python
# Neural network representation using Python data structures
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        import numpy as np
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))
    
    def forward(self, X):
        """Forward pass through the network"""
        activations = [X]
        for i in range(len(self.weights)):
            Z = activations[-1].dot(self.weights[i]) + self.biases[i]
            # ReLU activation for hidden layers, sigmoid for output
            if i < len(self.weights) - 1:
                A = np.maximum(0, Z)  # ReLU
            else:
                A = 1 / (1 + np.exp(-Z))  # Sigmoid
            activations.append(A)
        return activations
    
    def predict(self, X):
        """Predict class labels"""
        activations = self.forward(X)
        return (activations[-1] > 0.5).astype(int)
```

### Hyperparameter Tuning
```python
# Grid search implementation using Python data structures
def grid_search(X, y, model_class, param_grid, cv=5):
    """Simple grid search implementation"""
    import numpy as np
    from sklearn.model_selection import KFold
    
    # Generate all parameter combinations
    from itertools import product
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    # Store results
    results = []
    
    # Cross-validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    for params in param_combinations:
        param_dict = dict(zip(param_keys, params))
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Initialize and train model
            model = model_class(**param_dict)
            model.fit(X_train, y_train)
            
            # Evaluate
            score = model.score(X_val, y_val)
            cv_scores.append(score)
        
        # Store results
        results.append({
            'params': param_dict,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores)
        })
    
    # Sort by mean score
    results.sort(key=lambda x: x['mean_score'], reverse=True)
    return results
```

## Conclusion

This comprehensive guide covers the essential Python data structures you'll need for machine learning. By mastering these fundamentals, you'll be well-prepared to tackle the ML projects in your 30-day learning plan. Remember that effective data manipulation is the foundation of successful machine learning models.

As you progress through your learning journey, you'll find that these data structures become second nature, allowing you to focus more on algorithm implementation and model optimization rather than basic data handling.

Happy learning!
