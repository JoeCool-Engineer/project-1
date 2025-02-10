import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

def demonstrate_scaling_effects(save_path: Path = None):
    """Show how different scalings affect covariance structure
    
    Args:
        save_path: Optional path to save figure
    Returns:
        matplotlib figure object
    """
    # Generate correlated data
    X = np.random.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], size=1000)
    
    # Apply different scalings
    scales = [0.1, 1, 10]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, scale in enumerate(scales):
        X_scaled = X * [scale, 1]
        cov = np.cov(X_scaled.T)
        
        # Plot data points
        axes[i].scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.5)
        axes[i].set_title(f'Scale factor: {scale}\nCond. number: {np.linalg.cond(cov):.1f}')
        axes[i].axis('equal')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

def visualize_covariance_ellipse(X: np.ndarray, confidence: float = 0.95, save_path: Path = None):
    """Visualize covariance as confidence ellipse
    
    Args:
        X: Data matrix of shape (n_samples, 2)
        confidence: Confidence level for ellipse
        save_path: Optional path to save figure
    Returns:
        matplotlib figure object
    """
    # Compute mean and covariance
    mean = X.mean(axis=0)
    cov = np.cov(X.T)
    
    # Get eigenvalues and eigenvectors
    evals, evecs = np.linalg.eigh(cov)
    
    # Chi-square value for confidence level
    chi2_val = np.sqrt(2 * stats.chi2.ppf(confidence, df=2))
    
    # Generate ellipse points
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse = np.c_[np.cos(theta), np.sin(theta)]
    
    # Transform unit circle to ellipse
    transform = evecs @ np.diag(np.sqrt(evals))
    ellipse = chi2_val * (ellipse @ transform.T) + mean
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, label='Data points')
    ax.plot(ellipse[:, 0], ellipse[:, 1], 'r-', label=f'{confidence*100}% confidence')
    ax.axis('equal')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)
    return fig

def demonstrate_rank_deficiency(n_samples: int = 1000, save_path: Path = None):
    """Show how linear dependencies create rank deficiency
    
    Args:
        n_samples: Number of samples to generate
        save_path: Optional path to save figure
    Returns:
        matplotlib figure object
    """
    # Generate data with perfect correlation
    x = np.random.normal(0, 1, n_samples)
    X = np.c_[x, 2*x + np.random.normal(0, 0.01, n_samples)]
    
    # Compute covariance
    cov = np.cov(X.T)
    evals = np.linalg.eigvals(cov)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Data plot
    ax1.scatter(X[:, 0], X[:, 1], alpha=0.5)
    ax1.set_title('Nearly Dependent Features')
    ax1.axis('equal')
    
    # Eigenvalue plot
    ax2.bar(['λ1', 'λ2'], evals)
    ax2.set_title(f'Eigenvalues\nCondition number: {evals.max()/evals.min():.1f}')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

if __name__ == "__main__":
    # Create output directory
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    
    # Generate example figures
    demonstrate_scaling_effects(output_dir / "scaling_effects.png")
    
    # Generate data for covariance ellipse
    X = np.random.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], size=1000)
    visualize_covariance_ellipse(X, save_path=output_dir / "covariance_ellipse.png")
    
    demonstrate_rank_deficiency(save_path=output_dir / "rank_deficiency.png")
