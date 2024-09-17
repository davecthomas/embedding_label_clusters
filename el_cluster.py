from datetime import datetime
# import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from el_openai import ClusterNameStructuredOutput, ElOpenAI
from el_snowflake import ElSnowflake
from embedding_label import EmbeddingLabel
import hdbscan
import umap
import numpy as np
from typing import List, Dict, Any


from enum import Enum

# the number of samples to use from each cluster when querying the LLM
NUM_SAMPLES_PER_CLUSTER = 25
MAX_REVIEWS_TO_CLASSIFY = 100


class ClusteringManager:
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the ClusteringManager with a DataFrame of embeddings.

        Args:
            df (pd.DataFrame): DataFrame containing embeddings and metadata.
        """
        # Store the embeddings and texts directly as attributes
        self.df = df

        # Extract the embeddings as a NumPy array
        self.embeddings = ElOpenAI.convert_embeddings_to_numeric_array(df)

        # Ensure embeddings have the right dimensionality
        if self.embeddings.ndim != 2:
            raise ValueError(
                "Embeddings should be a 2D array where each row is an embedding vector.")

        # Store the original text (unused unless we are plotting)
        # self.texts = df['body'].tolist()

    def kmeans_clustering(self, reduced_embeddings, n_clusters: int = 7) -> List[int]:
        """
        Clusters embeddings using K-Means.
        This will cluster non-deterministically unless we set a fixed random state number
        In order for our clusters to be assigned the same way each time we run the code, 
        we need to set a fixed random state number.
        """
        ensure_deterministic_random_state: int = 42
        """Clusters embeddings using K-Means."""
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=ensure_deterministic_random_state)
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

    def reduce_with_pca(self, n_components: int = 100):
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

    # def plot_clusters(self, reduced_embeddings: np.ndarray, labels: List[int], title: str):
    #     """Plots the clusters after dimensionality reduction with appropriate labels and titles."""
    #     plt.figure(figsize=(10, 7))
    #     scatter = plt.scatter(
    #         reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='Spectral', s=50)
    #     plt.colorbar(scatter)
    #     plt.title(title)

    #     # Add some annotations to map dots to reviews
    #     for i in range(len(reduced_embeddings)):
    #         plt.annotate(self.texts[i][:20],  # Annotate the first 20 characters of the review
    #                      (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
    #                      fontsize=8, alpha=0.7)

    # def test_clustering_methods(self, reduced_embeddings):
    #     test_results = {}

    #     # K-Means clustering
    #     num_clusters: int = 7
    #     kmeans_labels = self.kmeans_clustering(
    #         reduced_embeddings, n_clusters=num_clusters)
    #     test_results['kmeans'] = kmeans_labels
    #     self.plot_clusters(reduced_embeddings,
    #                        kmeans_labels, "K-Means Clustering")

    #     # Assuming 'kmeans_labels' contains the cluster labels and 'texts' contains the corresponding comments
    #     for cluster_num in range(num_clusters):  # Loop over the clusters
    #         print(f"Cluster {cluster_num}:")
    #         count = 0  # To keep track of how many comments we've printed for each cluster
    #         for i, label in enumerate(kmeans_labels):
    #             if label == cluster_num:
    #                 print(f"  Comment: {self.texts[i]}")  # Print the comment
    #                 count += 1
    #                 if count >= 10:  # Stop after 10 comments
    #                     break

    #     # # DBSCAN clustering
    #     # dbscan_labels = self.dbscan_clustering(reduced_embeddings)
    #     # test_results['dbscan'] = dbscan_labels
    #     # self.plot_clusters(reduced_embeddings,
    #     #                    dbscan_labels, "DBSCAN Clustering")

    #     # # HDBSCAN clustering
    #     # hdbscan_labels = self.hdbscan_clustering(reduced_embeddings)
    #     # test_results['hdbscan'] = hdbscan_labels
    #     # self.plot_clusters(reduced_embeddings,
    #     #                    hdbscan_labels, "HDBSCAN Clustering")

    #     # # Agglomerative clustering
    #     # agglomerative_labels = self.agglomerative_clustering(
    #     #     reduced_embeddings)
    #     # test_results['agglomerative'] = agglomerative_labels
    #     # self.plot_clusters(reduced_embeddings,
    #     #                    agglomerative_labels, "Agglomerative Clustering")

    #     # Display all plots at once
    #     plt.show()

    #     return test_results

    def name_clusters_with_llm(self, df: pd.DataFrame, kmeans_labels: List[int], openai_client: ElOpenAI):
        """
        Name clusters by sampling comments from each cluster and asking the LLM to suggest a name.

        Args:
            df (pd.DataFrame): The DataFrame containing embeddings and metadata.
            kmeans_labels (List[int]): List of cluster labels.
            openai_client (ElOpenAI): The OpenAI client to use for querying the LLM.

        Returns:
            pd.DataFrame: The updated DataFrame with LLM-suggested cluster names and quality scores.
        """
        # Add the cluster numbers to the DataFrame
        df['cluster_num'] = kmeans_labels

        # Initialize dictionaries to store LLM-suggested cluster names and quality scores
        cluster_name_mapping = {}
        cluster_score_mapping = {}

        # Step 2: Get the model's token limit from the LLM token limits
        max_tokens = openai_client.get_llm_token_limit()

        # Step 3: Loop through each cluster and query the LLM for a cluster name and quality score
        unique_clusters = df['cluster_num'].unique()
        for cluster_num in unique_clusters:
            # Sample comments from the cluster (up to NUM_SAMPLES_PER_CLUSTER)
            cluster_sample = df[df['cluster_num'] == cluster_num].sample(
                min(NUM_SAMPLES_PER_CLUSTER, len(df[df['cluster_num'] == cluster_num])))
            list_texts = cluster_sample['body'].tolist()

            # Count tokens and check if we exceed the model's limit
            total_tokens = sum([openai_client.count_tokens(text)
                               for text in list_texts])
            print(f"Cluster {cluster_num}: Total Tokens = {total_tokens}")

            if total_tokens > max_tokens:
                print(f"Warning: Token limit exceeded for cluster {
                      cluster_num}. Reducing comments.")
                reduction_fraction = max_tokens / total_tokens
                reduced_num_samples = int(
                    NUM_SAMPLES_PER_CLUSTER * reduction_fraction)
                print(f"Reducing from {NUM_SAMPLES_PER_CLUSTER} comments to {
                      reduced_num_samples} comments.")
                list_texts = list_texts[:reduced_num_samples]

            # Query the LLM to suggest a name and quality score for the cluster
            cluster_name_result: ClusterNameStructuredOutput = openai_client.name_cluster(
                cluster_number=f"Cluster {cluster_num}",
                list_texts=list_texts
            )

            # Store the suggested name and quality score for the cluster
            cluster_name_mapping[cluster_num] = cluster_name_result.suggested_cluster_name
            cluster_score_mapping[cluster_num] = cluster_name_result.quality_score
            print(f"Cluster {cluster_num}: Suggested Name: {
                  cluster_name_result.suggested_cluster_name}; Quality Score: {cluster_name_result.quality_score}")

        # Step 4: Apply the LLM-suggested cluster names and quality scores to the DataFrame
        df['cluster_name'] = df['cluster_num'].map(cluster_name_mapping)
        df['quality_score'] = df['cluster_num'].map(cluster_score_mapping)

        # Return the updated DataFrame
        return df


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Append the timestamp to the filename
    filename = f"code_review_clusters_{timestamp}.csv"
    df: pd.DataFrame = pd.DataFrame()

    # Load the DataFrame from the CSV file (if it exists)
    if os.path.exists(filename):
        print(f"Loading embeddings from {filename}...")

        try:
            df = pd.read_csv(filename)

            # Convert 'embedding' string back to a list of floats (if necessary)
            df['embedding'] = df['embedding'].apply(lambda emb: eval(emb))

        except IOError as e:
            print(f"Error reading embeddings from CSV: {e}")
            exit(1)

        except Exception as e:
            print(f"An error occurred while loading embeddings: {e}")
            exit(1)

    # If no embeddings were loaded, generate new embeddings
    if df.empty:
        print(f"{filename} not found or empty. Generating new embeddings...")

        try:
            embedding_label = EmbeddingLabel()
            df: pd.DataFrame = embedding_label.generate_review_embeddings(
                limit=MAX_REVIEWS_TO_CLASSIFY)

            if df.empty:
                raise ValueError("No embeddings were generated.")

            print(f"New embeddings generated.")

        except Exception as e:
            print(f"An error occurred while generating embeddings: {e}")
            exit(1)

    # Initialize the ClusteringManager with the DataFrame
    manager = ClusteringManager(df)

    # Step 1: Reduce dimensionality to 50 dimensions for clustering
    reduced_embeddings = manager.reduce_with_pca(n_components=100)

    # Step 2: Run K-Means clustering and get the labels (array of integers)
    kmeans_labels = manager.kmeans_clustering(reduced_embeddings, n_clusters=7)

    # Initialize the OpenAI client
    openai_client = ElOpenAI()

    # Step 3: Name the clusters using LLM and update the DataFrame
    df = manager.name_clusters_with_llm(df, kmeans_labels, openai_client)

    # Step 4: Drop the 'user' column if it exists
    df = df.drop(columns=['user', 'dimensions'], errors='ignore')

    # df['embedding'] = df['embedding'].apply(lambda emb: str(emb))

    # Step 6: Save the updated DataFrame to CSV
    df.to_csv(filename, index=False)

    print(f"CSV successfully saved to {filename}.")
    # Step 7: Summarize how many reviews per cluster
    cluster_summary = df['cluster_name'].value_counts()
    print("\nCluster Review Summary:")
    print(cluster_summary)

    print("Storing the review classifications in Snowflake so we have a training data set...")
    snowflake_client = ElSnowflake()
    # Store the classifications and comments in Snowflake
    snowflake_client.store_classification(df)
    print("\nDone. Review classifications stored in Snowflake. Check the 'pr_review_comments_training' table.")
