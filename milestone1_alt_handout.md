# Milestone 1 (Alternate): Covariance Estimation and Analysis
**Due: Should be finished by 2/11/2025**

## Overview
In this milestone, you will implement and analyze covariance estimation from data, building directly on the concepts from Lessons 9 (Random Vectors) and 10 (Covariance Matrices). 

Choose one of the provided datasets:
1. Stock Returns (`stock_returns.csv`)
2. Sensor Readings (`sensor_readings.csv`)
3. Image Features (`image_features.csv`)

## Learning Objectives
1. Implement and validate covariance estimation from data
2. Understand the critical role of centering in covariance estimation
3. Visualize and interpret covariance structures
4. Analyze how centering affects statistical properties

## Required Deliverables

### 1. Implementation (40%)

#### Covariance Estimation
This function implements the core statistical concept of covariance estimation:
- Input: Matrix X where each row is an observation and each column is a variable
- Output: Square matrix showing relationships between all pairs of variables
- Key steps:
  1. Center the data (optional but important)
  2. Compute pairwise relationships
  3. Ensure result is symmetric

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

def estimate_covariance(X: np.ndarray, 
                       centered: bool = True) -> np.ndarray:
    """
    Compute sample covariance matrix from data
    
    Args:
        X: Data matrix (n_samples, n_features)
        centered: Whether to center the data first (subtract mean)
            - True: Compute proper covariance using (X - mean)
            - False: Use raw X (not recommended, included for comparison)
        
    Returns:
        Covariance matrix (n_features, n_features)
        
    Notes:
        Centering is crucial for covariance estimation because:
        1. Removes mean offset that would bias correlation estimates
        2. Makes results interpretable as variance around the mean
        3. Ensures positive semidefinite property of result
    """
    # TODO: Implement covariance estimation
    # Hints:
    # - Review Lesson 10 for covariance matrix definition
    # - Look up numpy's mean() function parameters
    # - Think about matrix multiplication with transpose
    # - Consider how to verify your result is symmetric
    pass
```

#### Estimator Comparison

This function validates your implementation against numpy's trusted version:

- Compare your results with numpy.cov()
- Check for numerical differences
- Understand any discrepancies
- Good software engineering practice: test against known good implementation
- It's ok if they are not exactly the same, but they should be close

```python
def compare_estimators(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compare your estimator to numpy's implementation
    
    Args:
        X: Data matrix
        
    Returns:
        Your covariance matrix, numpy's covariance matrix
    """
    # TODO: Compare your implementation to np.cov()
    pass
```

#### Sample Size Analysis

This function explores how many samples we need for reliable estimates:

- Try different sample sizes (small to large)
- Sampling means to draw samples from the dataset and compute only on the smaller sample
- See how estimates stabilize
- Important for real applications where data is limited
- Helps understand estimation uncertainty
- Advanced (repeatedly sample at the same sample size)

```python
def analyze_sample_size(X: np.ndarray,
                       sizes: Optional[np.ndarray] = None) -> dict:
    """
    Analyze how covariance estimate changes with sample size

    Args:
        X: Full dataset
        sizes: Array of sample sizes to test
        
    Returns:
        Dictionary with results
    """
    if sizes is None:
        sizes = np.logspace(1, np.log10(len(X)), 20).astype(int)
    
    # TODO: For each size n:
    # 1. Sample n points randomly
    # 2. Compute covariance
    # 3. Track how estimate changes
    pass
```

### 2. Analysis (40%)

#### Sample Size Effects

- Plot how covariance estimates converge as n increases
    - Hint: Plot Frobenius norm of difference between successive estimates
        - Frobenius norm is the sum of the squares of the elements of the matrix
        - You can use np.linalg.norm(A, ord='fro') to compute the Frobenius norm
    - Try logarithmic spacing of sample sizes
    - Look for where the curve flattens out
- Identify minimum sample size needed for stable estimates
    - Look for where changes between successive estimates become small
    - Consider setting a threshold (e.g., < 1% change)
    - Think about your application's accuracy needs
- Compare your results with numpy's implementation
    - Use np.allclose() with reasonable tolerance
    - Remember: Small numerical differences are normal
    - Focus on pattern similarity rather than exact matches

#### Visualization and Interpretation

- Create covariance heatmaps
- Plot confidence ellipses
- Interpret the meaning of:
    - Diagonal elements
    - Off-diagonal elements
    - Positive vs negative covariance

#### Understanding Data Preprocessing

- Centering Analysis:
    - Compare covariance with/without centering
    - Visualize how centering affects the data cloud
    - Prove mathematically why centering is necessary (optional)
- Basic Scaling Introduction:
    - Scaling means to divide each variable (column) by its standard deviation
    - Compare raw vs standardized variables
    - Show when different scales cause problems


### 3. Documentation (20%)

1. Implementation Details
   - Clear comments explaining your code
   - Proper docstrings
   - Example usage

2. Analysis Results
   - Clear figures with proper labels
   - Interpretation of results
   - Discussion of limitations

3. Reflection
   - What surprised you?
   - What was challenging?
   - What would you do differently?

## Optional Advanced Topics

### Population vs Sample Statistics

- Research the difference between population and sample statistics
- Investigate why numpy.cov() uses (n-1) denominator by default
- Compare with scipy.stats covariance functions
- Experiment with different denominators (n vs n-1)
- Consider when each might be appropriate

Example exploration:
```python
# Compare different covariance implementations
from scipy import stats

# Your implementation
cov_yours = estimate_covariance(X)

# NumPy implementation (uses n-1)
cov_numpy = np.cov(X.T)

# Manual calculation with n denominator
X_centered = X - X.mean(axis=0)
cov_pop = (X_centered.T @ X_centered) / len(X)

# Compare results and consider:
# - When do the differences matter?
# - Why might we prefer one over another?
# - What assumptions are we making?
```
