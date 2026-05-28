import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 1. DATA
def generate_gaussian_clusters(n=600, k=3, d=2, spread=0.6, separation=5):
    centers = np.random.randn(k, d) * separation
    X = []

    for c in centers:
        points = c + spread * np.random.randn(n // k, d)
        X.append(points)
    return np.vstack(X)

def normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

def normalize_safe(X):
    std = X.std(axis=0)
    std[std == 0] = 1  # prevent division by zero
    return (X - X.mean(axis=0)) / std

# 2. UTILITIES
def compute_distances(X, centroids):
    return np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)

def compute_inertia(X, centroids, labels):
    return np.sum((X - centroids[labels])**2)

# 3. INITIALIZATION
def random_init(X, k):
    idx = np.random.choice(X.shape[0], k, replace=False)
    return X[idx]

def kmeans_plus_plus_init(X, k):
    n = X.shape[0]
    centroids = []

    idx = np.random.randint(n)
    centroids.append(X[idx])

    for _ in range(1, k):
        distances = compute_distances(X, np.array(centroids))
        min_dist_sq = np.min(distances**2, axis=1)
        probs = min_dist_sq / np.sum(min_dist_sq)
        next_idx = np.random.choice(n, p=probs)
        centroids.append(X[next_idx])
    return np.array(centroids)

# 4. K-MEANS
def kmeans(X, k, init_centroids, max_iter=100, tol=1e-4):
    centroids = init_centroids.copy()
    history = []

    for _ in range(max_iter):
        distances = compute_distances(X, centroids)
        labels = np.argmin(distances, axis=1)

        inertia = compute_inertia(X, centroids, labels)
        history.append(inertia)

        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j)
            else centroids[j]
            for j in range(k)])

        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    return centroids, labels, inertia, history

# 5. SINGLE RUN
def run_kmeans_full(X, k, method):
    if method == "random":
        init = random_init(X, k)
    else:
        init = kmeans_plus_plus_init(X, k)

    centroids, labels, inertia, history = kmeans(X, k, init)
    return centroids, labels, inertia, history

# 6. MULTIPLE RUNS
def run_experiments_full(X, k, n_runs=50):
    results = {"random": [], "kmeans++": []}

    for _ in range(n_runs):
        results["random"].append(run_kmeans_full(X, k, "random"))
        results["kmeans++"].append(run_kmeans_full(X, k, "kmeans++"))
    return results

# 7. ANALYSIS
def extract_inertia(results):
    return {method: [run[2] for run in runs]
        for method, runs in results.items()}

def summarize(inertia_results):
    summary = {}

    for method, values in inertia_results.items():
        values = np.array(values)
        summary[method] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values)
        }
    return summary

def get_best_worst(results):
    output = {}

    for method, runs in results.items():
        runs_sorted = sorted(runs, key=lambda x: x[2])
        output[method] = {
            "best": runs_sorted[0],
            "worst": runs_sorted[-1]
        }
    return output

# 8. VISUALIZATION
def plot_histograms(inertia_results):
    plt.figure()
    for method, values in inertia_results.items():
        plt.hist(values, bins=15, alpha=0.5, label=method)

    plt.xlabel("Inertia")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Inertia Distribution")
    plt.show()

def plot_boxplot(inertia_results):
    plt.figure()
    data = [inertia_results["random"], inertia_results["kmeans++"]]

    plt.boxplot(data, labels=["Random", "k-means++"])
    plt.ylabel("Inertia")
    plt.title("Stability Comparison")
    plt.show()

def plot_runs(inertia_results):
    plt.figure()

    plt.plot(inertia_results["random"], label="Random", marker='o')
    plt.plot(inertia_results["kmeans++"], label="k-means++", marker='x')

    plt.xlabel("Run")
    plt.ylabel("Inertia")
    plt.title("Sensitivity to Initialization")
    plt.legend()
    plt.show()

def plot_clusters(X, labels, centroids, title):
    if X.shape[1] != 2:
        return

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=10)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100)
    plt.title(title)
    plt.show()

def plot_convergence(history, title):
    plt.figure()
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Inertia")
    plt.title(title)
    plt.show()

# 9. MAIN PIPELINE
if __name__ == "__main__":

    X = generate_gaussian_clusters(n=600, k=3, d=2)
    X = normalize(X)

    results = run_experiments_full(X, k=3, n_runs=50)
    inertia_results = extract_inertia(results)
    summary = summarize(inertia_results)

    print("\nSummary:")
    for method, stats in summary.items():
        print(method, stats)

    plot_histograms(inertia_results)
    plot_boxplot(inertia_results)
    plot_runs(inertia_results)

    bw = get_best_worst(results)

    for method in ["random", "kmeans++"]:
        best = bw[method]["best"]
        worst = bw[method]["worst"]

        plot_clusters(X, best[1], best[0], f"{method} - BEST")
        plot_clusters(X, worst[1], worst[0], f"{method} - WORST")

        plot_convergence(best[3], f"{method} - Convergence (BEST)")
        plot_convergence(worst[3], f"{method} - Convergence (WORST)")

from sklearn.datasets import load_digits

def load_digits_dataset():
    digits = load_digits()
    X = digits.data
    y = digits.target
    return X, y

if __name__ == "__main__":

    print("\nRunning experiments on handwritten digits dataset...")

    X_digits, y_digits = load_digits_dataset()
    X_digits = normalize_safe(X_digits)

    k = 10

    results_digits = run_experiments_full(X_digits, k=k, n_runs=50)
    inertia_digits = extract_inertia(results_digits)
    summary_digits = summarize(inertia_digits)

    print("\nDigits Dataset Summary:")
    for method, stats in summary_digits.items():
        print(method, stats)

    plot_histograms(inertia_digits)
    plot_boxplot(inertia_digits)
    plot_runs(inertia_digits)