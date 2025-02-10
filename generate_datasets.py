import numpy as np
from pathlib import Path
import pandas as pd

def generate_correlated_data(n_samples=1000, n_features=10, correlation=0.7, random_state=42):
    """Generate dataset with known correlation structure"""
    np.random.seed(random_state)
    
    # Create correlation matrix
    corr_matrix = np.eye(n_features)
    for i in range(n_features):
        for j in range(i+1, n_features):
            corr_matrix[i,j] = correlation ** abs(i-j)
            corr_matrix[j,i] = corr_matrix[i,j]
    
    # Generate data
    data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=corr_matrix,
        size=n_samples
    )
    
    # Create DataFrame
    columns = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=columns)
    return df

def generate_scale_varying_data(n_samples=1000, n_features=10, random_state=42):
    """Generate dataset with varying scales"""
    np.random.seed(random_state)
    
    # Generate base data
    data = np.random.randn(n_samples, n_features)
    
    # Apply different scales
    scales = np.logspace(-2, 2, n_features)
    data = data * scales[np.newaxis,:]
    
    # Create DataFrame
    columns = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=columns)
    return df

def generate_low_rank_data(n_samples=1000, n_features=10, rank=3, noise_level=0.1, random_state=42):
    """Generate low rank dataset with noise"""
    np.random.seed(random_state)
    
    # Generate low rank component
    U = np.random.randn(n_samples, rank)
    V = np.random.randn(rank, n_features)
    low_rank = U @ V
    
    # Add noise
    noise = noise_level * np.random.randn(n_samples, n_features)
    data = low_rank + noise
    
    # Create DataFrame
    columns = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=columns)
    return df

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent / "datasets"
    data_dir.mkdir(exist_ok=True)
    
    # Generate datasets
    datasets = {
        "stock_returns": generate_correlated_data(),  # Financial data with correlations
        "sensor_readings": generate_scale_varying_data(),  # IoT sensors with different units
        "image_features": generate_low_rank_data()  # Image compression-like data
    }
    
    # Save datasets
    for name, df in datasets.items():
        df.to_csv(data_dir / f"{name}_data.csv", index=False)
        
    print("Generated datasets:")
    for name in datasets:
        print(f"- {name}_data.csv")
