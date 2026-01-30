# Statistics Cheat Sheet for Machine Learning

A comprehensive guide to essential statistical concepts for machine learning practitioners.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Descriptive Statistics](#descriptive-statistics)
3. [Mean (Average)](#mean-average)
4. [Deviations](#deviations)
5. [Squared Deviations](#squared-deviations)
6. [Variance](#variance)
7. [Standard Deviation](#standard-deviation)
8. [Standard Error](#standard-error)
9. [Normalization](#normalization)
10. [Practical Examples](#practical-examples)

---

## Introduction

Statistics is the backbone of machine learning. Understanding these fundamental concepts will help you:
- Preprocess data effectively
- Understand model behavior
- Evaluate model performance
- Make informed decisions about feature engineering

This guide covers the essential statistical concepts you'll encounter in machine learning, with clear explanations and practical examples.

---

## Descriptive Statistics

**What is it?**  
Descriptive statistics summarize and describe the main features of a dataset. They provide simple summaries about the sample and measures.

**Why it matters in ML:**  
Before training any model, you need to understand your data's distribution, central tendency, and spread. This helps in:
- Detecting outliers
- Choosing appropriate algorithms
- Understanding feature importance
- Debugging model issues

---

## Mean (Average)

### Definition

The mean is the sum of all values divided by the number of values. It represents the central tendency of your data.

### Formula

For a dataset with n values:

$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i = \frac{x_1 + x_2 + ... + x_n}{n}$$

Where:
- x̄ (x-bar) is the sample mean
- n is the number of observations
- xᵢ is each individual value

### Example

Given house prices: [200k, 250k, 300k, 350k, 400k]

$$\bar{x} = \frac{200 + 250 + 300 + 350 + 400}{5} = \frac{1500}{5} = 300k$$

### Why it's useful in ML

- **Feature scaling baseline**: Used in normalization and standardization
- **Imputation**: Filling missing values with mean is a common strategy
- **Gradient descent**: The mean of gradients is used in batch training
- **Evaluation**: Mean Absolute Error (MAE) and Mean Squared Error (MSE) are key metrics

### Limitations

- Sensitive to outliers (one extreme value can skew the mean)
- May not represent the "typical" value in skewed distributions

---

## Deviations

### Definition

Deviation measures how far each data point is from the mean. It shows the "distance" of each observation from the average.

### Formula

For each data point:

$$\text{deviation}_i = x_i - \bar{x}$$

### Example

Using our house prices with mean = 300k:

| Price (xᵢ) | Deviation (xᵢ - x̄) |
|------------|---------------------|
| 200k | 200 - 300 = -100k |
| 250k | 250 - 300 = -50k |
| 300k | 300 - 300 = 0k |
| 350k | 350 - 300 = +50k |
| 400k | 400 - 300 = +100k |

**Important property**: The sum of all deviations always equals zero!

$$\sum_{i=1}^{n} (x_i - \bar{x}) = 0$$

### Why it's useful in ML

- **Understanding spread**: Shows how data points differ from average
- **Anomaly detection**: Large deviations indicate potential outliers
- **Feature engineering**: Creating deviation-based features
- **Residuals**: In regression, residuals are deviations from predicted values

---

## Squared Deviations

### Definition

Since deviations sum to zero, we square them to get all positive values. This amplifies larger deviations and is the foundation for variance.

### Formula

$$\text{squared deviation}_i = (x_i - \bar{x})^2$$

### Example

Continuing with house prices:

| Price | Deviation | Squared Deviation |
|-------|-----------|-------------------|
| 200k | -100k | 10,000 |
| 250k | -50k | 2,500 |
| 300k | 0k | 0 |
| 350k | +50k | 2,500 |
| 400k | +100k | 10,000 |

Sum of squared deviations: **25,000**

### Why it's useful in ML

- **Loss functions**: Mean Squared Error (MSE) is based on squared deviations
- **Penalizes outliers more**: Squaring gives more weight to large errors
- **Mathematical convenience**: Makes calculus operations easier
- **Least squares optimization**: Foundation of linear regression

---

## Variance

### Definition

Variance measures how spread out the data is from the mean. It's the average of squared deviations.

### Formulas

**Population variance** (when you have all data):

$$\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2$$

**Sample variance** (when estimating from a sample):

$$s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

Where:
- σ² (sigma squared) is population variance
- s² is sample variance
- n-1 is used for Bessel's correction (reduces bias in estimation)

### Example

Using our house prices (treating as sample):

$$s^2 = \frac{25,000}{5-1} = \frac{25,000}{4} = 6,250$$

Variance = 6,250 (in squared units: k²)

### Why it's useful in ML

- **Feature selection**: High variance features may contain more information
- **PCA**: Principal Component Analysis maximizes variance
- **Model evaluation**: Bias-variance tradeoff is crucial
- **Regularization**: Controlling variance helps prevent overfitting
- **Understanding data spread**: Low variance might indicate constant/uninformative features

---

## Standard Deviation

### Definition

Standard deviation is the square root of variance. It measures spread in the same units as the original data, making it more interpretable.

### Formula

**Population standard deviation**:

$$\sigma = \sqrt{\sigma^2} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2}$$

**Sample standard deviation**:

$$s = \sqrt{s^2} = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}$$

### Example

From our variance calculation:

$$s = \sqrt{6,250} \approx 79.06k$$

Standard deviation ≈ 79k (same units as original prices)

### Interpretation

For normally distributed data:
- ~68% of data falls within 1 standard deviation of mean
- ~95% within 2 standard deviations
- ~99.7% within 3 standard deviations

In our example:
- 68% of houses: 300k ± 79k = [221k, 379k]
- 95% of houses: 300k ± 158k = [142k, 458k]

### Why it's useful in ML

- **Feature scaling**: Standardization uses mean and standard deviation
- **Outlier detection**: Data points beyond 3σ are often considered outliers
- **Initialization**: Neural network weights often initialized using standard deviation
- **Batch normalization**: Normalizes using batch statistics
- **Understanding model confidence**: Standard deviation of predictions indicates uncertainty

---

## Standard Error

### Definition

Standard error measures the precision of a sample mean as an estimate of the population mean. It tells you how much your sample mean would vary if you repeated your sampling.

### Formula

$$SE = \frac{s}{\sqrt{n}} = \frac{\sigma}{\sqrt{n}}$$

Where:
- SE is the standard error
- s is the sample standard deviation
- n is the sample size

### Example

For our house prices with s = 79k and n = 5:

$$SE = \frac{79}{\sqrt{5}} = \frac{79}{2.236} \approx 35.3k$$

### Key Insight

**As sample size increases, standard error decreases**:

| Sample Size (n) | Standard Error |
|----------------|----------------|
| 5 | 35.3k |
| 25 | 15.8k |
| 100 | 7.9k |
| 400 | 3.95k |

Notice: To halve the error, you need 4× the data!

### Limits of Standard Error

The standard error approaches zero as sample size approaches infinity:

$$\lim_{n \to \infty} SE = \lim_{n \to \infty} \frac{s}{\sqrt{n}} = 0$$

This means with infinite data, your sample mean perfectly estimates the population mean.

### Confidence Intervals

Standard error is used to construct confidence intervals:

**95% Confidence Interval**:

$$\text{CI}_{95\%} = \bar{x} \pm 1.96 \times SE$$

For our example:

$$\text{CI}_{95\%} = 300 \pm 1.96 \times 35.3 = 300 \pm 69.2 = [230.8k, 369.2k]$$

### Why it's useful in ML

- **Model evaluation**: Understanding uncertainty in performance metrics
- **A/B testing**: Determining if model improvements are statistically significant
- **Cross-validation**: Standard error of CV scores indicates stability
- **Hyperparameter tuning**: Confidence in optimal parameter selection
- **Sample size determination**: How much data do you need for reliable estimates?
- **Bootstrapping**: Resampling methods use SE concepts

---

## Normalization

### Definition

Normalization transforms data to a common scale without distorting differences in ranges. This is crucial for many ML algorithms.

### Types of Normalization

#### 1. Min-Max Normalization (Scaling to [0,1])

$$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

**Example:**  
Original prices: [200k, 250k, 300k, 350k, 400k]

$$x_{\text{norm}} = \frac{x - 200}{400 - 200} = \frac{x - 200}{200}$$

| Original | Normalized |
|----------|-----------|
| 200k | 0.0 |
| 250k | 0.25 |
| 300k | 0.5 |
| 350k | 0.75 |
| 400k | 1.0 |

#### 2. Z-Score Standardization (Standard Scaling)

$$z = \frac{x - \bar{x}}{s}$$

**Example:**  
With mean = 300k and std = 79k:

$$z = \frac{x - 300}{79}$$

| Original | Standardized |
|----------|--------------|
| 200k | -1.27 |
| 250k | -0.63 |
| 300k | 0.0 |
| 350k | 0.63 |
| 400k | 1.27 |

#### 3. Robust Scaling (using median and IQR)

$$x_{\text{robust}} = \frac{x - \text{median}}{\text{IQR}}$$

Useful when data has outliers.

### When to Use Each Method

| Method | Use When | Examples |
|--------|----------|----------|
| Min-Max | Bounded range needed, no severe outliers | Neural networks, image processing |
| Z-Score | Gaussian distribution assumed | Linear regression, logistic regression, SVM |
| Robust | Data has outliers | Real-world datasets with anomalies |

### Why it's useful in ML

- **Gradient descent**: Converges faster with normalized features
- **Distance-based algorithms**: KNN, K-Means need same scale
- **Neural networks**: Better convergence and stability
- **Regularization**: Fair penalty across features (L1, L2)
- **Comparing features**: Different units become comparable
- **Preventing dominance**: High-magnitude features don't overshadow others

### Example Impact

Without normalization:
- Feature 1: House size (500-5000 sq ft)
- Feature 2: Number of bedrooms (1-5)

A distance calculation would be dominated by house size! Normalization fixes this.

---

## Practical Examples

### Example 1: Detecting Outliers in Housing Data

Given dataset: [200k, 210k, 205k, 215k, 900k]

$$\bar{x} = 346k, \quad s = 310k$$

The 900k house is:

$$z = \frac{900 - 346}{310} = 1.79$$

While not extreme (< 3σ), it's worth investigating!

### Example 2: Feature Scaling Before Training

**Before scaling:**
```
Features: [Income: $50,000, Age: 35, Credit Score: 720]
```

These have different scales! Neural networks will struggle.

**After Z-score standardization:**
```
Features: [Income: 0.23, Age: -0.45, Credit Score: 1.10]
```

Now all features contribute fairly to the model.

### Example 3: Evaluating Model Stability

Cross-validation results (accuracy): [0.85, 0.87, 0.84, 0.86, 0.88]

$$\bar{x} = 0.86, \quad s = 0.015, \quad SE = \frac{0.015}{\sqrt{5}} = 0.0067$$

**95% CI**: [0.86 - 1.96(0.0067), 0.86 + 1.96(0.0067)] = [0.847, 0.873]

Your model's true performance is likely between 84.7% and 87.3%.

### Example 4: Comparing Two Models

| Metric | Model A | Model B |
|--------|---------|---------|
| Mean Accuracy | 0.85 | 0.86 |
| Std Dev | 0.02 | 0.08 |

Model B has higher mean but much higher variance! Model A might be more reliable despite lower average performance.

---

## Quick Reference Table

| Concept | Formula | Units | When to Use |
|---------|---------|-------|-------------|
| Mean | x̄ = (1/n)Σxᵢ | Same as data | Central tendency |
| Deviation | xᵢ - x̄ | Same as data | Individual spread |
| Variance | s² = (1/(n-1))Σ(xᵢ-x̄)² | Squared units | Overall spread |
| Std Dev | s = √(s²) | Same as data | Interpretable spread |
| Std Error | SE = s/√n | Same as data | Estimate precision |
| Z-Score | z = (x-x̄)/s | Unitless | Standardization |

---

## Key Takeaways

1. **Mean** tells you the center of your data
2. **Variance/Standard Deviation** tells you how spread out it is
3. **Standard Error** tells you how confident you are in your estimates
4. **Normalization** puts everything on a level playing field
5. **Always visualize** your data alongside these statistics

---

## Further Learning

- **Next steps**: Probability distributions, hypothesis testing, correlation
- **Practice**: Use `pandas.describe()` to compute these statistics
- **Visualization**: Pair statistics with histograms and box plots

---

## Contributing

Feel free to submit issues or pull requests to improve this cheat sheet!

---

## License

MIT License - Free to use and modify

---

**Remember**: Statistics is not just math—it's a way of thinking about uncertainty and variation in data. Master these concepts, and you'll build better ML models! 🚀