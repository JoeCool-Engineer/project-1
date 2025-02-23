# Milestone 2 (Alternate): SVD Analysis and Component Selection
**Due: Should be finished by 2/18/2025**

## Overview
Building on your covariance analysis from Milestone 1, you'll now implement SVD-based analysis techniques and explore how to select important components. This connects directly to Lessons 11 (Feature Scaling) and 12 (SVD and Covariance).

Continue using your chosen dataset from Milestone 1:
1. Stock Returns (`stock_returns.csv`)
2. Sensor Readings (`sensor_readings.csv`)
3. Image Features (`image_features.csv`)

## Learning Objectives
1. Implement and understand SVD computation
2. Analyze explained variance through SVD
3. Compare different component selection methods
4. Visualize and interpret SVD results

## Required Deliverables

### 1. Implementation (40%)

# SVD Computation
The Singular Value Decomposition (SVD) is a fundamental matrix factorization:
- Decomposes data matrix X into U, s, and Vh
- U: New coordinate system based on data variation
- s: Importance of each direction (singular values)
- Vh: How original features combine into new directions

Key preprocessing choices:
- Centering: Remove mean to focus on variation
- Scaling: Make features comparable when units differ
- Both affect interpretation of results

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler

def compute_svd(X: np.ndarray, 
                center: bool = True,
                scale: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute SVD of data matrix with optional preprocessing
    
    Args:
        X: Data matrix (n_samples, n_features)
        center: Whether to center the data
        scale: Whether to scale to unit variance
        
    Returns:
        U: Left singular vectors (n_samples, n_features)
           - Rows are observations
           - Columns are new coordinate directions
        s: Singular values in descending order
           - Square roots of eigenvalues
           - Measure importance of each direction
        Vh: Right singular vectors (n_features, n_features)
           - Rows are principal directions
           - Columns are original features
    """
    # TODO: Implement SVD computation
    # Hints:
    # - Remember centering from Milestone 1
    # - Consider when scaling helps/hurts interpretation
    # - Look up numpy.linalg.svd full_matrices parameter
    # - Think about numerical stability with small values
    pass
```

# Explained Variance Analysis
This function quantifies how much variation each component captures:
- Square singular values to get variances
- Convert to percentages of total variance
- Cumulative sum shows total explained variance
- Helps decide how many components to keep

```python
def analyze_explained_variance(s: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Analyze explained variance from singular values
    
    Args:
        s: Array of singular values
        
    Returns:
        Dictionary with:
        - explained_variance_ratio
        - cumulative_variance_ratio
    """
    # TODO: Compute variance ratios
    pass
```

# Component Selection
This function implements automated dimension selection:
- Uses explained variance to choose components
- threshold parameter sets minimum variance to retain
- Higher threshold keeps more components
- Lower threshold gives more aggressive reduction
- Balance between complexity and information retention

```python
def select_components(s: np.ndarray,
                     threshold: float = 0.95) -> int:
    """
    Select number of components using variance threshold
    
    Args:
        s: Array of singular values
        threshold: Minimum cumulative variance to explain
        
    Returns:
        Number of components to keep
    """
    # TODO: Implement component selection
    pass
```

### 2. Analysis (40%)

#### SVD Analysis
1. For a simple 2×2 matrix (choose one):
   ```
   A = [2 1]    or    B = [3 0]    or    C = [1 1]
       [1 2]          [0 2]             [1 0]
   ```
   - Compute SVD by hand showing all steps
   - Verify your result multiplying U·Σ·V^T
   - Compare with numpy.linalg.svd output

2. For your chosen dataset:
   - Apply your SVD implementation
   - Analyze effect of preprocessing
   - Interpret U, s, and Vh matrices

#### Component Selection
- Implement and compare methods:
  * Variance threshold (e.g., keep 90% of variance)
  * Elbow method (plot variance vs components)
  * Kaiser criterion (keep components with σᵢ > mean(σ))
- Justify your recommendations

#### Visualization
- Create scree plots
- Plot cumulative variance
- Visualize component coefficients

### 3. Documentation (20%)

#### Implementation Notes
- Clear explanation of your SVD implementation
- Document preprocessing choices
- Note any numerical considerations

#### Analysis Results
- Show your hand calculations
- Compare with computational results
- Explain any differences

#### Future Work
- What would you do differently?
- How could you improve stability?
- What other applications interest you?

