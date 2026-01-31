# Machine Learning Math Cheat Sheet 📐

> A beginner-friendly guide to essential mathematical concepts in Machine Learning

## 📚 Table of Contents

1. [Why Math Matters in ML](#why-math-matters-in-ml)
2. [Linear Regression](#1-linear-regression)
3. [Statistics Toolkit](#2-statistics-toolkit)
4. [Vectors](#3-vectors)
5. [Matrices](#4-matrices)
6. [Putting It All Together](#5-putting-it-all-together)
7. [Quick Reference](#quick-reference)

---

## Why Math Matters in ML

Machine Learning is **applied mathematics**. Here's what each concept does:

| Concept | What It Does | Why You Need It |
|---------|--------------|-----------------|
| **Linear Regression** | Finds patterns in data | Foundation of all ML models |
| **Statistics** | Measures and understands data | Data preprocessing & evaluation |
| **Vectors** | Represents data points | How computers see data |
| **Matrices** | Processes multiple data points | Efficient computation |

**The Big Picture:**
```
Raw Data → Vectors → Matrices → Linear Regression → Predictions
              ↓           ↓             ↓
          Statistics helps at every step
```

---

# 1. Linear Regression

## What Is It?

Finding the best line (or hyperplane) that fits your data points.

```
y = w * x + b
```

- `y` = output (what we predict)
- `x` = input (what we know)
- `w` = weight/slope (how much y changes with x)
- `b` = bias/intercept (starting point)

## Why We Need It

✅ **Foundation of ML** - All neural networks are built from linear functions  
✅ **Interpretable** - Easy to understand and explain  
✅ **Fast** - Quick to train and predict  
✅ **Baseline** - Starting point for any ML problem  

## Types of Linear Regression

### Simple Linear Regression
One input, one output:
```
y = w * x + b
```

**Example:** Predict house price from size
```python
x = 1500  # square feet
w = 200   # dollars per sq ft
b = 50000 # base price

y = 200 * 1500 + 50000 = 350,000  # predicted price
```

### Multiple Linear Regression
Many inputs, one output:
```
y = w1*x1 + w2*x2 + w3*x3 + ... + b
```

**Example:** Predict house price from size, bedrooms, age
```python
x1 = 1500   # square feet
x2 = 3      # bedrooms
x3 = 10     # age in years

w1 = 200    # weight for size
w2 = 10000  # weight for bedrooms
w3 = -2000  # weight for age (negative = older = cheaper)
b = 50000   # base price

y = 200*1500 + 10000*3 + (-2000)*10 + 50000
y = 300000 + 30000 - 20000 + 50000 = 360,000
```

## The Cost Function (Loss)

Measures how wrong our predictions are:

### Mean Squared Error (MSE)
```
MSE = (1/n) * Σ(yi - y_hat_i)²
```

**Step-by-step calculation:**
```python
# True values
y = [100, 200, 300, 400]

# Predictions
y_hat = [110, 190, 310, 390]

# Step 1: Calculate errors
errors = [100-110, 200-190, 300-310, 400-390]
       = [-10, 10, -10, 10]

# Step 2: Square the errors
squared_errors = [100, 100, 100, 100]

# Step 3: Average them
MSE = (100 + 100 + 100 + 100) / 4 = 100
```

### Root Mean Squared Error (RMSE)
```
RMSE = sqrt(MSE)
```

**Why use RMSE?**
- Same units as the output
- Easier to interpret

```python
RMSE = sqrt(100) = 10
# This means our predictions are off by ~10 on average
```

### Mean Absolute Error (MAE)
```
MAE = (1/n) * Σ|yi - y_hat_i|
```

Less sensitive to outliers than MSE:
```python
errors = [-10, 10, -10, 10]
absolute_errors = [10, 10, 10, 10]
MAE = (10 + 10 + 10 + 10) / 4 = 10
```

## Gradient Descent

How we find the best w and b:

```
w_new = w_old - learning_rate * (∂Loss/∂w)
b_new = b_old - learning_rate * (∂Loss/∂b)
```

**For MSE loss:**
```
∂Loss/∂w = -(2/n) * Σ[xi * (yi - y_hat_i)]
∂Loss/∂b = -(2/n) * Σ(yi - y_hat_i)
```

**Step-by-step example:**
```python
# Data
x = [1, 2, 3, 4]
y = [2, 4, 6, 8]

# Initial guess
w = 0
b = 0
learning_rate = 0.1

# Iteration 1:
y_hat = [0, 0, 0, 0]  # predictions with w=0, b=0
errors = [2, 4, 6, 8]

# Compute gradients
dw = -(2/4) * (1*2 + 2*4 + 3*6 + 4*8) = -(2/4) * 60 = -30
db = -(2/4) * (2 + 4 + 6 + 8) = -(2/4) * 20 = -10

# Update parameters
w = 0 - 0.1 * (-30) = 3
b = 0 - 0.1 * (-10) = 1

# Iteration 2:
y_hat = [3*1+1, 3*2+1, 3*3+1, 3*4+1] = [4, 7, 10, 13]
# Continue until convergence...
```

## When to Use Linear Regression

✅ **Use when:**
- Relationship between input and output is roughly linear
- You need interpretability
- You have numerical outputs (not categories)
- Fast training is important

❌ **Don't use when:**
- Relationship is highly non-linear
- You need to classify categories
- Data has complex patterns

---

# 2. Statistics Toolkit

## Basic Measures

### Mean (Average)
```
mean = (Σ xi) / n
```

**Example:**
```python
data = [10, 20, 30, 40, 50]
mean = (10 + 20 + 30 + 40 + 50) / 5 = 30
```

**Why it matters:** Centers your data, used in normalization

### Median (Middle Value)
```
median = middle value when sorted
```

**Example:**
```python
data = [10, 20, 100, 30, 40]
sorted_data = [10, 20, 30, 40, 100]
median = 30  # middle value
```

**Why it matters:** Robust to outliers (100 doesn't skew it)

### Mode (Most Frequent)
```
mode = most common value
```

**Example:**
```python
data = [1, 2, 2, 3, 4, 2, 5]
mode = 2  # appears 3 times
```

**Why it matters:** Useful for categorical data

### Variance
Measures spread of data:
```
variance = (1/n) * Σ(xi - mean)²
```

**Step-by-step:**
```python
data = [10, 20, 30, 40, 50]
mean = 30

# Deviations from mean
deviations = [10-30, 20-30, 30-30, 40-30, 50-30]
           = [-20, -10, 0, 10, 20]

# Squared deviations
squared = [400, 100, 0, 100, 400]

# Average
variance = (400 + 100 + 0 + 100 + 400) / 5 = 200
```

### Standard Deviation
```
std = sqrt(variance)
```

**Example:**
```python
std = sqrt(200) ≈ 14.14
```

**Interpretation:** Data typically varies by ±14.14 from the mean

**Why it matters:** 
- Normalization (z-score)
- Understanding data spread
- Detecting outliers

## Normalization Techniques

### Min-Max Normalization
Scales data to [0, 1]:
```
x_norm = (x - min(x)) / (max(x) - min(x))
```

**Example:**
```python
data = [10, 20, 30, 40, 50]
min_val = 10
max_val = 50

x = 30
x_norm = (30 - 10) / (50 - 10) = 20 / 40 = 0.5

# All values:
[10, 20, 30, 40, 50] → [0, 0.25, 0.5, 0.75, 1.0]
```

### Z-Score Normalization (Standardization)
Centers around 0 with std of 1:
```
z = (x - mean) / std
```

**Example:**
```python
data = [10, 20, 30, 40, 50]
mean = 30
std = 14.14

x = 40
z = (40 - 30) / 14.14 ≈ 0.71

# Interpretation: x is 0.71 standard deviations above mean
```

**When to use which:**
- **Min-Max:** When you need specific range [0,1], neural networks
- **Z-Score:** When you want to preserve distribution shape, comparing different scales

## Correlation

Measures relationship between two variables:
```
correlation = Σ[(xi - x_mean) * (yi - y_mean)] / (n * std_x * std_y)
```

**Range:** -1 to +1

- `+1` = perfect positive correlation (both increase together)
- `0` = no correlation
- `-1` = perfect negative correlation (one increases, other decreases)

**Example:**
```python
# Study hours vs test scores
hours = [1, 2, 3, 4, 5]
scores = [50, 60, 70, 80, 90]

# Strong positive correlation ≈ 1.0
# More study → higher scores
```

**Why it matters:**
- Feature selection (remove highly correlated features)
- Understanding relationships
- Detecting multicollinearity

## Probability Distributions

### Normal Distribution (Gaussian)
```
Bell curve: f(x) = (1 / (σ * sqrt(2π))) * e^(-(x-μ)²/(2σ²))
```

**Properties:**
- Mean = μ (mu)
- Standard deviation = σ (sigma)
- 68% of data within ±1σ
- 95% of data within ±2σ
- 99.7% of data within ±3σ

**Why it matters:** Many ML algorithms assume normally distributed data

### Bernoulli Distribution
Binary outcomes (0 or 1):
```
P(x=1) = p
P(x=0) = 1-p
```

**Example:** Coin flip, classification (spam/not spam)

## Common Statistical Tests

### Hypothesis Testing
```
1. Null hypothesis (H0): No effect/difference
2. Alternative hypothesis (H1): There is an effect
3. Calculate p-value
4. If p < 0.05: Reject H0 (statistically significant)
```

**Why it matters:** Validating ML model improvements

---

# 3. Vectors

## What Is a Vector?

An ordered list of numbers representing a point in space:
```
v = [v1, v2, v3, ..., vn]
```

**In ML:** Each data point is a vector!

**Example:**
```python
# A house represented as a vector
house = [1500, 3, 10, 1]
#        [sqft, beds, age, garage]
```

## Vector Operations

### Vector Addition
Add corresponding elements:
```
a + b = [a1+b1, a2+b2, a3+b3]
```

**Example:**
```python
a = [1, 2, 3]
b = [4, 5, 6]
a + b = [1+4, 2+5, 3+6] = [5, 7, 9]
```

**Why it matters:** Combining features, gradient updates

### Scalar Multiplication
Multiply every element by a number:
```
c * v = [c*v1, c*v2, c*v3]
```

**Example:**
```python
v = [1, 2, 3]
3 * v = [3, 6, 9]
```

**Why it matters:** Scaling data, learning rates

### Dot Product (Inner Product)
Multiply corresponding elements and sum:
```
a · b = a1*b1 + a2*b2 + a3*b3
```

**Example:**
```python
a = [1, 2, 3]
b = [4, 5, 6]
a · b = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
```

**Geometric meaning:**
```
a · b = |a| * |b| * cos(θ)
```
Where θ is the angle between vectors

**Special cases:**
- If a · b = 0 → vectors are perpendicular
- If a · b > 0 → vectors point in similar directions
- If a · b < 0 → vectors point in opposite directions

**Why it matters:** 
- Core of linear regression: `y = w · x + b`
- Neural networks
- Similarity measures

### Vector Length (Magnitude/Norm)
```
|v| = sqrt(v1² + v2² + v3²)
```

**Example:**
```python
v = [3, 4]
|v| = sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5
```

**Why it matters:** 
- Normalization
- Distance calculations
- Regularization

## Types of Norms

### L1 Norm (Manhattan Distance)
```
||v||₁ = |v1| + |v2| + |v3|
```

**Example:**
```python
v = [3, -4, 5]
||v||₁ = |3| + |-4| + |5| = 3 + 4 + 5 = 12
```

### L2 Norm (Euclidean Distance)
```
||v||₂ = sqrt(v1² + v2² + v3²)
```

**Example:**
```python
v = [3, 4]
||v||₂ = sqrt(9 + 16) = 5
```

**Why norms matter:**
- Different regularization techniques (L1, L2)
- Distance metrics
- Normalization strategies

## Distance Metrics

### Euclidean Distance
Straight-line distance:
```
distance = sqrt((x1-y1)² + (x2-y2)² + (x3-y3)²)
```

**Example:**
```python
point_a = [1, 2]
point_b = [4, 6]

distance = sqrt((1-4)² + (2-6)²)
         = sqrt(9 + 16)
         = sqrt(25) = 5
```

### Manhattan Distance
Sum of absolute differences:
```
distance = |x1-y1| + |x2-y2| + |x3-y3|
```

**Example:**
```python
point_a = [1, 2]
point_b = [4, 6]

distance = |1-4| + |2-6| = 3 + 4 = 7
```

### Cosine Similarity
Measures angle between vectors:
```
similarity = (a · b) / (|a| * |b|)
```

**Range:** -1 to +1
- +1 = same direction
- 0 = perpendicular
- -1 = opposite direction

**Example:**
```python
a = [1, 2, 3]
b = [2, 4, 6]  # same direction, different magnitude

a · b = 1*2 + 2*4 + 3*6 = 28
|a| = sqrt(1 + 4 + 9) = sqrt(14)
|b| = sqrt(4 + 16 + 36) = sqrt(56)

similarity = 28 / (sqrt(14) * sqrt(56)) = 1.0
```

**Why it matters:**
- Text similarity (document comparison)
- Recommendation systems
- Clustering

## Unit Vectors

Vector with length 1:
```
u = v / |v|
```

**Example:**
```python
v = [3, 4]
|v| = 5
u = [3/5, 4/5] = [0.6, 0.8]

# Check: |u| = sqrt(0.36 + 0.64) = 1 ✓
```

**Why it matters:** Directional information without magnitude

---

# 4. Matrices

## What Is a Matrix?

A 2D array of numbers (table of data):
```
A = [ a11  a12  a13 ]
    [ a21  a22  a23 ]
    [ a31  a32  a33 ]
```

**Dimensions:** m × n (m rows, n columns)

**In ML:** 
- Each row = one data sample
- Each column = one feature

**Example:**
```python
# Dataset of 3 houses
houses = [ [1500, 3, 10],    # house 1: sqft, beds, age
           [2000, 4, 5],     # house 2
           [1200, 2, 15] ]   # house 3

# This is a 3×3 matrix
```

## Matrix Operations

### Matrix Addition
Add corresponding elements (matrices must be same size):
```
A + B = [ a11+b11  a12+b12 ]
        [ a21+b21  a22+b22 ]
```

**Example:**
```python
A = [ [1, 2],      B = [ [5, 6],
      [3, 4] ]           [7, 8] ]

A + B = [ [6, 8],
          [10, 12] ]
```

### Scalar Multiplication
Multiply every element:
```
c * A = [ c*a11  c*a12 ]
        [ c*a21  c*a22 ]
```

**Example:**
```python
A = [ [1, 2],
      [3, 4] ]

2 * A = [ [2, 4],
          [6, 8] ]
```

### Matrix Multiplication
**Rule:** (m × n) matrix × (n × p) matrix = (m × p) matrix

```
For C = A × B:
cij = Σ(aik * bkj)  for all k
```

**Step-by-step example:**
```python
A = [ [1, 2],      # 2×2 matrix
      [3, 4] ]

B = [ [5, 6],      # 2×2 matrix
      [7, 8] ]

# C[0,0] = 1*5 + 2*7 = 5 + 14 = 19
# C[0,1] = 1*6 + 2*8 = 6 + 16 = 22
# C[1,0] = 3*5 + 4*7 = 15 + 28 = 43
# C[1,1] = 3*6 + 4*8 = 18 + 32 = 50

C = [ [19, 22],
      [43, 50] ]
```

**Matrix-Vector multiplication:**
```python
A = [ [1, 2, 3],    # 2×3 matrix
      [4, 5, 6] ]

x = [7, 8, 9]       # 3×1 vector

A × x = [ 1*7 + 2*8 + 3*9,    = [50,
          4*7 + 5*8 + 6*9 ]     122]
```

**Why it matters:** This IS linear regression!
```
y = W × x + b
```

### Transpose
Flip rows and columns:
```
If A = [ [1, 2, 3],      then A^T = [ [1, 4],
         [4, 5, 6] ]                    [2, 5],
                                         [3, 6] ]
```

**Example:**
```python
A = [ [1, 2],
      [3, 4],
      [5, 6] ]    # 3×2 matrix

A^T = [ [1, 3, 5],
        [2, 4, 6] ]    # 2×3 matrix
```

**Why it matters:**
- Changing data orientation
- Matrix multiplication compatibility
- Gradient calculations

## Special Matrices

### Identity Matrix (I)
Diagonal of 1s, rest 0s:
```
I = [ [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1] ]
```

**Property:** A × I = I × A = A

**Why it matters:** Like multiplying by 1

### Zero Matrix
All elements are 0:
```
0 = [ [0, 0],
      [0, 0] ]
```

### Diagonal Matrix
Non-zero only on diagonal:
```
D = [ [d1, 0,  0 ],
      [0,  d2, 0 ],
      [0,  0,  d3] ]
```

**Why it matters:** Efficient computations, scaling

## Matrix Inverse

The matrix that "undoes" another:
```
A × A^(-1) = I
```

**Only exists if matrix is square and determinant ≠ 0**

**Example (2×2 inverse):**
```
A = [ [a, b],      A^(-1) = (1/det) * [ [d,  -b],
      [c, d] ]                          [-c,  a] ]

where det = ad - bc
```

**Concrete example:**
```python
A = [ [4, 7],
      [2, 6] ]

det = 4*6 - 7*2 = 24 - 14 = 10

A^(-1) = (1/10) * [ [6,  -7],
                     [-2,  4] ]
       
       = [ [0.6,  -0.7],
           [-0.2,  0.4] ]

# Verify: A × A^(-1) = I ✓
```

**Why it matters:** Solving linear systems (Ax = b → x = A^(-1)b)

## Determinant

Single number that describes a matrix:
```
For 2×2: det(A) = ad - bc
```

**Example:**
```python
A = [ [3, 8],
      [4, 6] ]

det(A) = 3*6 - 8*4 = 18 - 32 = -14
```

**Interpretation:**
- det = 0 → matrix is singular (no inverse)
- det ≠ 0 → matrix is invertible
- |det| = scaling factor for area/volume

## Eigenvalues and Eigenvectors

Special vectors that only get scaled when multiplied by a matrix:
```
A × v = λ × v
```

- v = eigenvector (direction that doesn't change)
- λ (lambda) = eigenvalue (how much it scales)

**Example:**
```python
A = [ [2, 1],
      [1, 2] ]

# Eigenvector: v = [1, 1]
# Eigenvalue: λ = 3

A × [1, 1] = [2+1, 1+2] = [3, 3] = 3 × [1, 1] ✓
```

**Why it matters:**
- Principal Component Analysis (PCA)
- Understanding data structure
- Dimensionality reduction

## Reshaping Operations

### Flatten
Convert matrix to vector:
```python
A = [ [1, 2, 3],
      [4, 5, 6] ]

flatten(A) = [1, 2, 3, 4, 5, 6]
```

### Reshape
Change dimensions (keeping total elements):
```python
v = [1, 2, 3, 4, 5, 6]

reshape(v, 2, 3) = [ [1, 2, 3],
                      [4, 5, 6] ]

reshape(v, 3, 2) = [ [1, 2],
                      [3, 4],
                      [5, 6] ]
```

**Why it matters:** Preparing data for different layers in neural networks

---

# 5. Putting It All Together

## Linear Regression with Matrix Notation

### The Matrix Form
```
Y = X × W + b
```

Where:
- Y = predictions (n × 1)
- X = input data (n × m)
- W = weights (m × 1)
- b = bias (scalar or n × 1)

**Concrete example:**
```python
# 3 houses, 2 features each
X = [ [1500, 3],    # house 1: sqft, bedrooms
      [2000, 4],    # house 2
      [1200, 2] ]   # house 3

# Weights for each feature
W = [ [200],        # price per sqft
      [10000] ]     # price per bedroom

b = 50000          # base price

# Matrix multiplication
Y = X × W + b

# House 1: 1500*200 + 3*10000 + 50000 = 380,000
# House 2: 2000*200 + 4*10000 + 50000 = 490,000
# House 3: 1200*200 + 2*10000 + 50000 = 310,000

Y = [ [380000],
      [490000],
      [310000] ]
```

### Gradient Descent in Matrix Form

**Loss (MSE):**
```
L = (1/n) * ||Y - X×W||²
```

**Gradient:**
```
∂L/∂W = -(2/n) * X^T × (Y - Y_hat)
```

**Update:**
```
W_new = W_old - learning_rate * ∂L/∂W
```

**Why matrix form is better:**
- Process all samples at once
- Much faster computation
- Leverage optimized libraries (NumPy, GPU)

## Real-World ML Pipeline

```
1. DATA COLLECTION
   └─> Raw data as matrix

2. PREPROCESSING (Statistics)
   ├─> Calculate mean, std
   ├─> Normalize using z-score
   └─> Handle missing values

3. FEATURE ENGINEERING (Vectors)
   ├─> Each sample is a vector
   ├─> Calculate distances
   └─> Compute similarities

4. MODEL TRAINING (Matrices + Linear Regression)
   ├─> Organize as matrix X
   ├─> Initialize weights W
   ├─> Compute Y = X × W
   └─> Update W using gradients

5. EVALUATION (Statistics)
   ├─> Calculate MSE, RMSE
   ├─> Compute correlation
   └─> Test statistical significance

6. PREDICTION (Matrix multiplication)
   └─> Y_new = X_new × W_learned
```

## Example: Complete Workflow

### Problem
Predict house prices from size and bedrooms.

### Step 1: Data (Matrix)
```python
# 5 houses
X = [ [1500, 3],
      [2000, 4],
      [1200, 2],
      [1800, 3],
      [1600, 3] ]

Y = [ [300000],
      [400000],
      [250000],
      [350000],
      [320000] ]
```

### Step 2: Statistics (Normalize)
```python
# Calculate mean and std for each feature
mean_sqft = 1620
std_sqft = 283.7

mean_beds = 3
std_beds = 0.71

# Z-score normalization
X_norm = [ [(1500-1620)/283.7, (3-3)/0.71],
           [(2000-1620)/283.7, (4-3)/0.71],
           ... ]

# Also normalize Y
mean_price = 324000
std_price = 56789
```

### Step 3: Initialize (Vector)
```python
W = [ [0],      # weight for sqft
      [0] ]     # weight for beds

b = 0
learning_rate = 0.01
```

### Step 4: Train (Matrix operations)
```python
for epoch in range(1000):
    # Forward pass (matrix multiplication)
    Y_hat = X_norm × W + b
    
    # Loss (vector operations)
    errors = Y_norm - Y_hat
    loss = mean(errors²)
    
    # Gradients (matrix operations)
    dW = -(2/n) * X_norm^T × errors
    
    # Update (vector operations)
    W = W - learning_rate * dW
    b = b - learning_rate * mean(errors)
```

### Step 5: Predict (Matrix multiplication)
```python
# New house: 1700 sqft, 3 bedrooms
X_new = [ [(1700-1620)/283.7, (3-3)/0.71] ]

Y_pred_norm = X_new × W + b

# Denormalize
Y_pred = Y_pred_norm * std_price + mean_price
```

## Common Patterns

### Pattern 1: Data as Matrices
```
Samples × Features = Matrix
  100   ×    5     = 100×5 matrix
```

### Pattern 2: Weights as Vectors
```
Features × 1 = Weight vector
   5     × 1 = 5×1 vector
```

### Pattern 3: Predictions via Multiplication
```
(Samples × Features) × (Features × 1) = Samples × 1
     100  ×    5     ×     5     × 1  =   100   × 1
```

---

# Quick Reference

## Linear Regression Formulas

```
Model:           y = w·x + b
Multiple:        y = w1×x1 + w2×x2 + ... + b
Matrix form:     Y = X × W + b

Loss (MSE):      L = (1/n) × Σ(yi - y_hat_i)²
Gradient:        ∂L/∂w = -(2/n) × Σ[xi(yi - y_hat_i)]
Update:          w_new = w_old - α × ∂L/∂w
```

## Statistics Formulas

```
Mean:            μ = (Σxi) / n
Variance:        σ² = (1/n) × Σ(xi - μ)²
Std Dev:         σ = sqrt(σ²)

Min-Max:         x_norm = (x - min) / (max - min)
Z-Score:         z = (x - μ) / σ

Correlation:     r = Σ[(xi-x̄)(yi-ȳ)] / (n × σx × σy)
```

## Vector Formulas

```
Addition:        a + b = [a1+b1, a2+b2, ...]
Scalar mult:     c × v = [c×v1, c×v2, ...]
Dot product:     a·b = a1×b1 + a2×b2 + ...
Length:          |v| = sqrt(v1² + v2² + ...)

L1 norm:         ||v||₁ = |v1| + |v2| + ...
L2 norm:         ||v||₂ = sqrt(v1² + v2² + ...)

Euclidean:       d = sqrt((x1-y1)² + (x2-y2)²)
Cosine sim:      cos(θ) = (a·b) / (|a| × |b|)
```

## Matrix Formulas

```
Addition:        [A + B]ij = aij + bij
Scalar mult:     [cA]ij = c × aij
Multiplication:  [AB]ij = Σk(aik × bkj)
Transpose:       [A^T]ij = aji

Identity:        I × A = A × I = A
Inverse:         A × A^(-1) = I
Determinant:     det([a,b; c,d]) = ad - bc
```

## When to Use What

| Task | Use | Example |
|------|-----|---------|
| Predict continuous value | Linear Regression | House prices, temperature |
| Understand data spread | Mean, Std, Variance | Is data consistent? |
| Scale features | Normalization | Prepare for training |
| Represent data point | Vector | [height, weight, age] |
| Store dataset | Matrix | All samples together |
| Compute predictions | Matrix multiplication | Y = X × W |
| Measure similarity | Cosine similarity | Document matching |
| Reduce dimensions | Eigenvalues/vectors | PCA |

## Common Mistakes to Avoid

❌ **Forgetting to normalize** → Slow/unstable training  
❌ **Wrong matrix dimensions** → Multiplication fails  
❌ **Using mean when outliers exist** → Use median  
❌ **Not checking for correlation** → Redundant features  
❌ **Dividing by zero** → Check determinant before inverse  
❌ **Wrong loss function** → MSE for regression, cross-entropy for classification  

## Learning Path

```
1. Start here:
   ├─> Vectors (how to represent data)
   ├─> Statistics (understand your data)
   └─> Linear regression (make predictions)

2. Then learn:
   ├─> Matrices (efficient computation)
   ├─> Gradient descent (optimization)
   └─> Feature engineering (better inputs)

3. Next steps:
   ├─> Multiple features
   ├─> Regularization (L1, L2)
   ├─> Polynomial regression
   └─> Logistic regression

4. Advanced:
   ├─> Neural networks
   ├─> Deep learning
   └─> Specialized models
```

## Essential Python Libraries

```python
import numpy as np           # Vectors and matrices
import pandas as pd          # Data manipulation
import matplotlib.pyplot as plt  # Visualization
from sklearn.linear_model import LinearRegression  # Pre-built models
from sklearn.preprocessing import StandardScaler   # Normalization
```

**Quick NumPy examples:**
```python
# Vectors
v = np.array([1, 2, 3])
w = np.array([4, 5, 6])
dot_product = np.dot(v, w)
length = np.linalg.norm(v)

# Matrices
X = np.array([[1, 2], [3, 4]])
W = np.array([[5], [6]])
Y = np.dot(X, W)               # Matrix multiplication
X_T = X.T                       # Transpose

# Statistics
mean = np.mean(data)
std = np.std(data)
normalized = (data - mean) / std
```

---

## Next Steps

After mastering these basics:

1. **Implement everything from scratch** - Best way to learn
2. **Then use libraries** - Understand what they do under the hood
3. **Work on projects** - Apply to real problems
4. **Learn advanced topics** - Neural networks, deep learning
5. **Keep practicing** - Math skills improve with use

---

## Additional Resources

- **Khan Academy** - Linear algebra, statistics foundations
- **3Blue1Brown** - Visual understanding of linear algebra
- **StatQuest** - Statistics explained simply
- **Fast.ai** - Practical deep learning
- **Andrew Ng's ML Course** - Comprehensive ML foundations

---

**Remember:** Every ML expert started exactly where you are now. The math might seem intimidating, but it's just:
- **Vectors** = lists of numbers
- **Matrices** = tables of numbers  
- **Statistics** = understanding patterns
- **Linear Regression** = finding the best line

Master these, and you have the foundation for all of machine learning! 🚀

---

*This cheat sheet is designed for beginners. Bookmark it, reference it often, and don't hesitate to work through the examples step by step!*