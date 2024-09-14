import matplotlib.pyplot as plt
import csv
import os
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from el_openai import ElOpenAI
from embedding_label import EmbeddingLabel
import hdbscan
import umap
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt


class ClusteringManager:
    def __init__(self, embeddings: List[Dict[str, Any]]):
        """
        Initializes the ClusteringManager with a list of embeddings and converts them to a NumPy array.

        Args:
            embeddings (List[Dict[str, Any]]): List of dictionaries or lists containing embeddings.
        """
        # Use the utility method from ElOpenAI to convert embeddings to a NumPy array
        self.embeddings = ElOpenAI.convert_embeddings_to_numeric_array(
            embeddings)

        # Ensure embeddings have the right dimensionality
        if self.embeddings.ndim != 2:
            raise ValueError(
                "Embeddings should be a 2D array where each row is an embedding vector.")

        # Store the original texts for labeling the plot
        self.texts = [e['text'] for e in embeddings]

    def kmeans_clustering(self, reduced_embeddings, n_clusters: int = 7) -> List[int]:
        """Clusters embeddings using K-Means."""
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(reduced_embeddings)
        return labels

    def dbscan_clustering(self, reduced_embeddings, eps: float = 0.5, min_samples: int = 5) -> List[int]:
        """Clusters embeddings using DBSCAN."""
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(reduced_embeddings)
        return labels

    def hdbscan_clustering(self, reduced_embeddings, min_cluster_size: int = 5) -> List[int]:
        """Clusters embeddings using HDBSCAN."""
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(reduced_embeddings)
        return labels

    def agglomerative_clustering(self, reduced_embeddings, n_clusters: int = 5) -> List[int]:
        """Clusters embeddings using Agglomerative Clustering."""
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agglomerative.fit_predict(reduced_embeddings)
        return labels

    def reduce_with_pca(self, n_components: int = 50):
        """Reduce dimensionality using PCA for clustering."""
        """
        # PCA Dimensionality Reduction Summary

        This step applies Principal Component Analysis (PCA) to reduce the dimensionality of the input embeddings.

        1. **Fitting the PCA model**:
        - PCA identifies the principal components (new axes) that capture the most variance in the dataset.
        - The principal components are determined by computing the covariance matrix and performing eigenvalue decomposition to find the directions in which the data varies the most.

        2. **Transforming the data**:
        - Once the principal components are identified, the original high-dimensional data is projected onto these new axes.
        - This reduces the number of dimensions while retaining the most important features of the data, which helps preserve its structure and meaning.

        3. **Purpose**:
        - Dimensionality reduction is critical for improving the efficiency of downstream tasks like clustering and visualization, as high-dimensional data is more computationally expensive and may suffer from the "curse of dimensionality."
        - By reducing the dimensionality (e.g., from 1536 to 50), we ensure that the most important patterns in the data are kept while eliminating noise and redundant information.

        4. **Result**:
        - The resulting `reduced_embeddings` have fewer dimensions (e.g., 50 instead of 1536) but still capture most of the variance, making them suitable for tasks like clustering (K-Means, HDBSCAN) or visualization.
        """

        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(self.embeddings)
        return reduced_embeddings

    def reduce_with_umap(self, n_components: int = 50):
        """Reduce dimensionality using UMAP for clustering."""
        reducer = umap.UMAP(n_components=n_components)
        reduced_embeddings = reducer.fit_transform(self.embeddings)
        return reduced_embeddings

    def plot_clusters(self, reduced_embeddings: np.ndarray, labels: List[int], title: str):
        """Plots the clusters after dimensionality reduction with appropriate labels and titles."""
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='Spectral', s=50)
        plt.colorbar(scatter)
        plt.title(title)

        # Add some annotations to map dots to reviews
        for i in range(len(reduced_embeddings)):
            plt.annotate(self.texts[i][:20],  # Annotate the first 20 characters of the review
                         (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                         fontsize=8, alpha=0.7)

    def test_clustering_methods(self, reduced_embeddings):
        test_results = {}

        # K-Means clustering
        num_clusters: int = 7
        kmeans_labels = self.kmeans_clustering(
            reduced_embeddings, n_clusters=num_clusters)
        test_results['kmeans'] = kmeans_labels
        self.plot_clusters(reduced_embeddings,
                           kmeans_labels, "K-Means Clustering")

        # Assuming 'kmeans_labels' contains the cluster labels and 'texts' contains the corresponding comments
        for cluster_num in range(num_clusters):  # Loop over the clusters
            print(f"Cluster {cluster_num}:")
            count = 0  # To keep track of how many comments we've printed for each cluster
            for i, label in enumerate(kmeans_labels):
                if label == cluster_num:
                    print(f"  Comment: {self.texts[i]}")  # Print the comment
                    count += 1
                    if count >= 10:  # Stop after 10 comments
                        break

        # # DBSCAN clustering
        # dbscan_labels = self.dbscan_clustering(reduced_embeddings)
        # test_results['dbscan'] = dbscan_labels
        # self.plot_clusters(reduced_embeddings,
        #                    dbscan_labels, "DBSCAN Clustering")

        # # HDBSCAN clustering
        # hdbscan_labels = self.hdbscan_clustering(reduced_embeddings)
        # test_results['hdbscan'] = hdbscan_labels
        # self.plot_clusters(reduced_embeddings,
        #                    hdbscan_labels, "HDBSCAN Clustering")

        # # Agglomerative clustering
        # agglomerative_labels = self.agglomerative_clustering(
        #     reduced_embeddings)
        # test_results['agglomerative'] = agglomerative_labels
        # self.plot_clusters(reduced_embeddings,
        #                    agglomerative_labels, "Agglomerative Clustering")

        # Display all plots at once
        plt.show()

        return test_results


if __name__ == "__main__":
    filename = "review_embeddings.csv"

    embeddings = []

    # Check if the CSV file exists
    if os.path.exists(filename):
        print(f"Loading embeddings from {filename}...")

        try:
            # Load embeddings from CSV
            with open(filename, mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Convert embedding string back to a list of floats
                    embedding_list = eval(row['embedding'])
                    embeddings.append({
                        "embedding": embedding_list,
                        "text": row['text'],
                        "dimensions": int(row['dimensions']),
                        "user": row['user']
                    })

            if len(embeddings) == 0:
                print(f"No embeddings found in {
                      filename}. Proceeding to generate new embeddings.")

        except IOError as e:
            print(f"Error reading embeddings from CSV: {e}")
            exit(1)

        except Exception as e:
            print(f"An error occurred while loading embeddings: {e}")
            exit(1)

    # If no embeddings were loaded from the file, generate new embeddings
    if len(embeddings) == 0:
        print(f"{filename} not found or empty. Generating new embeddings...")

        try:
            # Assume EmbeddingLabel is a class that generates embeddings
            embedding_label: EmbeddingLabel = EmbeddingLabel()
            embeddings = embedding_label.generate_review_embeddings(limit=100)

            if len(embeddings) == 0:
                raise ValueError("No embeddings were generated.")

            # Save the new embeddings to a CSV file
            ElOpenAI.save_to_csv(embeddings=embeddings, filename=filename)
            print(f"New embeddings saved to {filename}.")

        except Exception as e:
            print(f"An error occurred while generating embeddings: {e}")
            exit(1)

    # Initialize the ClusteringManager with embeddings
    manager = ClusteringManager(embeddings)

    # Step 1: Reduce dimensionality to 50 dimensions for clustering
    # or manager.reduce_with_umap(n_components=50)
    reduced_embeddings = manager.reduce_with_pca(n_components=50)

    # Step 2: Run the test method to check different clustering algorithms on reduced embeddings
    test_results = manager.test_clustering_methods(reduced_embeddings)

    # Print the clustering results
    for method, labels in test_results.items():
        print(f"{method} clustering results: {labels}")

    # Step 3: Further reduce dimensionality to 2 dimensions for plotting
    # or manager.reduce_with_umap(n_components=2)
    reduced_embeddings_2d = manager.reduce_with_pca(n_components=2)

    # Step 4: Plot clusters for each clustering method
    for method, labels in test_results.items():
        manager.plot_clusters(reduced_embeddings_2d, labels, f"{
                              method.capitalize()} Clustering")
