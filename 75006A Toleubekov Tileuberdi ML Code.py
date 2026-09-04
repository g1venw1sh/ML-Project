import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class KMeans:
    """Standard K-Means with random initialization"""
    
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.n_iterations = None
        
    def _initialize_centroids(self, X):
        """Random initialization: randomly select k points as centroids"""
        np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices].copy()
    
    def _assign_clusters(self, X):
        """Assign each point to nearest centroid"""
        distances = np.sqrt(((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        """Update centroids as mean of assigned points"""
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0)
            else:
                # Handle empty cluster by reinitializing to a random point
                new_centroids[k] = X[np.random.choice(X.shape[0])]
        return new_centroids
    
    def _compute_inertia(self, X, labels):
        """Compute sum of squared distances to centroids"""
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[k]) ** 2)
        return inertia
    
    def fit(self, X):
        """Fit K-Means to data"""
        self.centroids = self._initialize_centroids(X)
        
        for i in range(self.max_iters):
            # Assign clusters
            labels = self._assign_clusters(X)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check convergence
            centroid_shift = np.sum((new_centroids - self.centroids) ** 2)
            self.centroids = new_centroids
            
            if centroid_shift < self.tol:
                break
        
        self.labels = self._assign_clusters(X)
        self.inertia = self._compute_inertia(X, self.labels)
        self.n_iterations = i + 1
        return self
    
    def predict(self, X):
        """Assign new points to clusters"""
        return self._assign_clusters(X)

class KMeansPlusPlus:
    """K-Means with k-means++ initialization"""
    
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.n_iterations = None
    
    def _initialize_centroids(self, X):
        """K-means++ initialization"""
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        # Choose first centroid randomly
        centroids = [X[np.random.choice(n_samples)]]
        
        # Choose remaining centroids
        for _ in range(1, self.n_clusters):
            # Compute distances to nearest centroid
            distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in centroids]) 
                                 for x in X])
            
            # Choose next centroid with probability proportional to distance squared
            probs = distances / distances.sum()
            next_centroid_idx = np.random.choice(n_samples, p=probs)
            centroids.append(X[next_centroid_idx])
        
        return np.array(centroids)
    
    def _assign_clusters(self, X):
        """Assign each point to nearest centroid"""
        distances = np.sqrt(((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        """Update centroids as mean of assigned points"""
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0)
            else:
                new_centroids[k] = X[np.random.choice(X.shape[0])]
        return new_centroids
    
    def _compute_inertia(self, X, labels):
        """Compute sum of squared distances to centroids"""
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[k]) ** 2)
        return inertia
    
    def fit(self, X):
        """Fit K-Means++ to data"""
        self.centroids = self._initialize_centroids(X)
        
        for i in range(self.max_iters):
            # Assign clusters
            labels = self._assign_clusters(X)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check convergence
            centroid_shift = np.sum((new_centroids - self.centroids) ** 2)
            self.centroids = new_centroids
            
            if centroid_shift < self.tol:
                break
        
        self.labels = self._assign_clusters(X)
        self.inertia = self._compute_inertia(X, self.labels)
        self.n_iterations = i + 1
        
        return self
    
    def predict(self, X):
        """Assign new points to clusters"""
        return self._assign_clusters(X)


def run_experiment(X, n_clusters=3, n_trials=50, dataset_name="Dataset"):
    """Run multiple trials of K-Means and K-Means++ and collect results"""
    
    results = {
        'kmeans': {'inertia': [], 'iterations': [], 'labels': []},
        'kmeans_pp': {'inertia': [], 'iterations': [], 'labels': []}
    }
    
    for trial in range(n_trials):
        # Standard K-Means
        km = KMeans(n_clusters=n_clusters, random_state=trial)
        km.fit(X)
        results['kmeans']['inertia'].append(km.inertia)
        results['kmeans']['iterations'].append(km.n_iterations)
        results['kmeans']['labels'].append(km.labels)
        
        # K-Means++
        kmpp = KMeansPlusPlus(n_clusters=n_clusters, random_state=trial)
        kmpp.fit(X)
        results['kmeans_pp']['inertia'].append(kmpp.inertia)
        results['kmeans_pp']['iterations'].append(kmpp.n_iterations)
        results['kmeans_pp']['labels'].append(kmpp.labels)
    
    return results


def analyze_results(results, dataset_name):
    """Analyze and visualize experimental results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'K-Means vs K-Means++ on {dataset_name}', fontsize=14)
    
    # 1. Inertia distribution
    ax = axes[0, 0]
    ax.hist(results['kmeans']['inertia'], bins=20, alpha=0.5, label='K-Means', color='blue')
    ax.hist(results['kmeans_pp']['inertia'], bins=20, alpha=0.5, label='K-Means++', color='red')
    ax.set_xlabel('Inertia')
    ax.set_ylabel('Frequency')
    ax.set_title('Inertia Distribution')
    ax.legend()
    ax.axvline(np.min(results['kmeans_pp']['inertia']), color='red', linestyle='--', alpha=0.7)
    
    # 2. Convergence iterations
    ax = axes[0, 1]
    ax.hist(results['kmeans']['iterations'], bins=20, alpha=0.5, label='K-Means', color='blue')
    ax.hist(results['kmeans_pp']['iterations'], bins=20, alpha=0.5, label='K-Means++', color='red')
    ax.set_xlabel('Iterations to Converge')
    ax.set_ylabel('Frequency')
    ax.set_title('Convergence Speed')
    ax.legend()
    
    # 3. Box plot of inertia
    ax = axes[0, 2]
    data = [results['kmeans']['inertia'], results['kmeans_pp']['inertia']]
    bp = ax.boxplot(data, labels=['K-Means', 'K-Means++'])
    ax.set_ylabel('Inertia')
    ax.set_title('Inertia Box Plot')
    
    # 4. Scatter plot of final cluster assignments (best run)
    ax = axes[1, 0]
    best_kmeans_idx = np.argmin(results['kmeans']['inertia'])
    best_pp_idx = np.argmin(results['kmeans_pp']['inertia'])
    ax.scatter(range(len(results['kmeans']['inertia'])), 
              results['kmeans']['inertia'], alpha=0.6, label='K-Means', color='blue')
    ax.scatter(range(len(results['kmeans_pp']['inertia'])), 
              results['kmeans_pp']['inertia'], alpha=0.6, label='K-Means++', color='red')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Inertia')
    ax.set_title('Inertia Across Trials')
    ax.legend()
    
    # 5. Stability analysis (pairwise label similarity)
    ax = axes[1, 1]
    
    def compute_stability(labels_list):
        n = len(labels_list)
        similarities = []
        for i in range(n):
            for j in range(i+1, n):
                # Normalized mutual information (simplified as agreement rate)
                agreement = np.mean(labels_list[i] == labels_list[j])
                similarities.append(agreement)
        return similarities
    
    km_similarities = compute_stability(results['kmeans']['labels'])
    kmpp_similarities = compute_stability(results['kmeans_pp']['labels'])
    
    ax.hist(km_similarities, bins=20, alpha=0.5, label='K-Means', color='blue')
    ax.hist(kmpp_similarities, bins=20, alpha=0.5, label='K-Means++', color='red')
    ax.set_xlabel('Pairwise Label Agreement')
    ax.set_ylabel('Frequency')
    ax.set_title('Stability (Label Consistency)')
    ax.legend()
    
    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = f"""
    Summary Statistics:
    
    K-Means:
      Mean Inertia: {np.mean(results['kmeans']['inertia']):.2f}
      Std Inertia: {np.std(results['kmeans']['inertia']):.2f}
      Min Inertia: {np.min(results['kmeans']['inertia']):.2f}
      Mean Iterations: {np.mean(results['kmeans']['iterations']):.2f}
    
    K-Means++:
      Mean Inertia: {np.mean(results['kmeans_pp']['inertia']):.2f}
      Std Inertia: {np.std(results['kmeans_pp']['inertia']):.2f}
      Min Inertia: {np.min(results['kmeans_pp']['inertia']):.2f}
      Mean Iterations: {np.mean(results['kmeans_pp']['iterations']):.2f}
    
    Improvement:
      Inertia Reduction: {((np.mean(results['kmeans']['inertia']) - np.mean(results['kmeans_pp']['inertia'])) / np.mean(results['kmeans']['inertia']) * 100):.2f}%
      Std Reduction: {((np.std(results['kmeans']['inertia']) - np.std(results['kmeans_pp']['inertia'])) / np.std(results['kmeans']['inertia']) * 100):.2f}%
    """
    ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Generate synthetic dataset with good clustering potential
def generate_synthetic_data(n_samples=300, n_features=2, n_clusters=5, random_state=42):
    """Generate synthetic dataset with clear cluster structure"""
    X, y_true = make_blobs(n_samples=n_samples, 
                           n_features=n_features,
                           centers=n_clusters,
                           cluster_std=0.8,
                           random_state=random_state)
    return X, y_true

# Generate synthetic dataset with overlapping clusters
def generate_overlapping_data(n_samples=300, n_features=2, n_clusters=4, random_state=42):
    """Generate synthetic dataset with overlapping clusters"""
    X, y_true = make_blobs(n_samples=n_samples, 
                           n_features=n_features,
                           centers=n_clusters,
                           cluster_std=2.5,
                           random_state=random_state)
    return X, y_true

# Load MNIST dataset (subsampled for efficiency)
def load_mnist_subset(n_samples=1000, random_state=42):
    """Load a subset of MNIST dataset"""
    from sklearn.datasets import fetch_openml
    
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    
    # Subsample
    np.random.seed(random_state)
    indices = np.random.choice(X.shape[0], n_samples, replace=False)
    X_subset = X[indices]
    y_subset = y[indices]
    
    # Normalize
    X_subset = X_subset / 255.0
    
    # Apply PCA for visualization and dimensionality reduction
    pca = PCA(n_components=50, random_state=random_state)
    X_pca = pca.fit_transform(X_subset)
    
    print(f"MNIST subset shape: {X_pca.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    
    return X_pca, y_subset


# Run experiments
print("=" * 60)
print("EXPERIMENT 1: Synthetic Data with Clear Clusters")
print("=" * 60)

X_synthetic, y_true = generate_synthetic_data(n_samples=300, n_features=2, n_clusters=5)
results_synthetic = run_experiment(X_synthetic, n_clusters=5, n_trials=50, dataset_name="Synthetic")
analyze_results(results_synthetic, "Synthetic Data (Clear Clusters)")

print("\n" + "=" * 60)
print("EXPERIMENT 2: Synthetic Data with Overlapping Clusters")
print("=" * 60)

X_overlapping, y_true_overlap = generate_overlapping_data(n_samples=300, n_features=2, n_clusters=4)
results_overlapping = run_experiment(X_overlapping, n_clusters=4, n_trials=50, dataset_name="Overlapping")
analyze_results(results_overlapping, "Synthetic Data (Overlapping Clusters)")

print("\n" + "=" * 60)
print("EXPERIMENT 3: MNIST Dataset (High-Dimensional)")
print("=" * 60)

X_mnist, y_mnist = load_mnist_subset(n_samples=1000)
results_mnist = run_experiment(X_mnist, n_clusters=10, n_trials=30, dataset_name="MNIST")
analyze_results(results_mnist, "MNIST (High-Dimensional)")

#================

def visualize_clustering_comparison(X, y_true, n_clusters, dataset_name):
    """Visualize clustering results for 2D data"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Clustering Results Comparison - {dataset_name}', fontsize=14)
    
    # True labels
    ax = axes[0, 0]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=30)
    ax.set_title('True Labels')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    # Best K-Means
    km_best = KMeans(n_clusters=n_clusters, random_state=np.argmin(results_synthetic['kmeans']['inertia']))
    km_best.fit(X)
    ax = axes[0, 1]
    ax.scatter(X[:, 0], X[:, 1], c=km_best.labels, cmap='viridis', s=30)
    ax.scatter(km_best.centroids[:, 0], km_best.centroids[:, 1], 
              marker='x', s=200, linewidths=3, color='red', label='Centroids')
    ax.set_title(f'Best K-Means (Inertia: {km_best.inertia:.2f})')
    ax.legend()
    
    # Best K-Means++
    kmpp_best = KMeansPlusPlus(n_clusters=n_clusters, random_state=np.argmin(results_synthetic['kmeans_pp']['inertia']))
    kmpp_best.fit(X)
    ax = axes[0, 2]
    ax.scatter(X[:, 0], X[:, 1], c=kmpp_best.labels, cmap='viridis', s=30)
    ax.scatter(kmpp_best.centroids[:, 0], kmpp_best.centroids[:, 1], 
              marker='x', s=200, linewidths=3, color='red', label='Centroids')
    ax.set_title(f'Best K-Means++ (Inertia: {kmpp_best.inertia:.2f})')
    ax.legend()
    
    # Worst K-Means
    km_worst = KMeans(n_clusters=n_clusters, random_state=np.argmax(results_synthetic['kmeans']['inertia']))
    km_worst.fit(X)
    ax = axes[1, 0]
    ax.scatter(X[:, 0], X[:, 1], c=km_worst.labels, cmap='viridis', s=30)
    ax.scatter(km_worst.centroids[:, 0], km_worst.centroids[:, 1], 
              marker='x', s=200, linewidths=3, color='red', label='Centroids')
    ax.set_title(f'Worst K-Means (Inertia: {km_worst.inertia:.2f})')
    ax.legend()
    
    # Worst K-Means++
    kmpp_worst = KMeansPlusPlus(n_clusters=n_clusters, random_state=np.argmax(results_synthetic['kmeans_pp']['inertia']))
    kmpp_worst.fit(X)
    ax = axes[1, 1]
    ax.scatter(X[:, 0], X[:, 1], c=kmpp_worst.labels, cmap='viridis', s=30)
    ax.scatter(kmpp_worst.centroids[:, 0], kmpp_worst.centroids[:, 1], 
              marker='x', s=200, linewidths=3, color='red', label='Centroids')
    ax.set_title(f'Worst K-Means++ (Inertia: {kmpp_worst.inertia:.2f})')
    ax.legend()
    
    # Initial centroids comparison
    ax = axes[1, 2]
    ax.scatter(X[:, 0], X[:, 1], c='lightgray', s=30, alpha=0.5)
    
    # K-Means initial centroids
    km_initial = KMeans(n_clusters=n_clusters, random_state=0)
    km_init_centroids = km_initial._initialize_centroids(X)
    ax.scatter(km_init_centroids[:, 0], km_init_centroids[:, 1], 
              marker='o', s=100, color='blue', label='K-Means Init', alpha=0.7)
    
    # K-Means++ initial centroids
    kmpp_initial = KMeansPlusPlus(n_clusters=n_clusters, random_state=0)
    kmpp_init_centroids = kmpp_initial._initialize_centroids(X)
    ax.scatter(kmpp_init_centroids[:, 0], kmpp_init_centroids[:, 1], 
              marker='s', s=100, color='red', label='K-Means++ Init', alpha=0.7)
    ax.set_title('Initialization Comparison')
    ax.legend()
    
    plt.tight_layout()
    plt.show()


# Visualize for synthetic dataset
visualize_clustering_comparison(X_synthetic, y_true, n_clusters=5, 
                               dataset_name="Synthetic (Clear Clusters)")

# Additional analysis for overlapping clusters
visualize_clustering_comparison(X_overlapping, y_true_overlap, n_clusters=4, 
                               dataset_name="Synthetic (Overlapping Clusters)")

#=========================

def statistical_analysis(results, dataset_name):
    """Perform statistical analysis on clustering results"""
    
    print(f"\nStatistical Analysis for {dataset_name}")
    print("-" * 50)
    
    km_inertia = np.array(results['kmeans']['inertia'])
    kmpp_inertia = np.array(results['kmeans_pp']['inertia'])
    
    # Basic statistics
    print("K-Means:")
    print(f"  Mean inertia: {km_inertia.mean():.2f} ± {km_inertia.std():.2f}")
    print(f"  Median inertia: {np.median(km_inertia):.2f}")
    print(f"  Range: [{km_inertia.min():.2f}, {km_inertia.max():.2f}]")
    print(f"  Coefficient of variation: {km_inertia.std() / km_inertia.mean():.3f}")
    
    print("\nK-Means++:")
    print(f"  Mean inertia: {kmpp_inertia.mean():.2f} ± {kmpp_inertia.std():.2f}")
    print(f"  Median inertia: {np.median(kmpp_inertia):.2f}")
    print(f"  Range: [{kmpp_inertia.min():.2f}, {kmpp_inertia.max():.2f}]")
    print(f"  Coefficient of variation: {kmpp_inertia.std() / kmpp_inertia.mean():.3f}")
    
    # Improvement metrics
    mean_improvement = (km_inertia.mean() - kmpp_inertia.mean()) / km_inertia.mean() * 100
    std_improvement = (km_inertia.std() - kmpp_inertia.std()) / km_inertia.std() * 100
    
    print(f"\nImprovements:")
    print(f"  Mean inertia reduction: {mean_improvement:.2f}%")
    print(f"  Standard deviation reduction: {std_improvement:.2f}%")
    
    # Probability of finding good solution
    best_km = km_inertia.min()
    best_kmpp = kmpp_inertia.min()
    threshold = best_kmpp * 1.1  # Within 10% of best K-Means++ solution
    
    prob_km = np.mean(km_inertia <= threshold)
    prob_kmpp = np.mean(kmpp_inertia <= threshold)
    
    print(f"\nProbability of finding solution within 10% of best (threshold={threshold:.2f}):")
    print(f"  K-Means: {prob_km:.2%}")
    print(f"  K-Means++: {prob_kmpp:.2%}")
    
    # Convergence analysis
    km_iters = np.array(results['kmeans']['iterations'])
    kmpp_iters = np.array(results['kmeans_pp']['iterations'])
    
    print(f"\nConvergence:")
    print(f"  K-Means mean iterations: {km_iters.mean():.2f} ± {km_iters.std():.2f}")
    print(f"  K-Means++ mean iterations: {kmpp_iters.mean():.2f} ± {kmpp_iters.std():.2f}")
    
    return {
        'km_mean': km_inertia.mean(),
        'km_std': km_inertia.std(),
        'kmpp_mean': kmpp_inertia.mean(),
        'kmpp_std': kmpp_inertia.std(),
        'mean_improvement': mean_improvement,
        'std_improvement': std_improvement,
        'prob_km': prob_km,
        'prob_kmpp': prob_kmpp
    }


# Run statistical analysis
stats_synthetic = statistical_analysis(results_synthetic, "Synthetic Data (Clear Clusters)")
stats_overlapping = statistical_analysis(results_overlapping, "Synthetic Data (Overlapping Clusters)")
stats_mnist = statistical_analysis(results_mnist, "MNIST (High-Dimensional)")