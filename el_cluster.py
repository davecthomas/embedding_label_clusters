import matplotlib.pyplot as plt
import pandas as pd
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

from enum import Enum

# Define an Enum to represent the new cluster labels


class ClusterName(Enum):
    UI_UX_TESTING = "UI/UX Testing and Technical Adjustments"
    CODE_REFACTORING = "Code Refactoring and Conventions"
    POSITIVE_FEEDBACK = "Positive Feedback and Follow-ups"
    MINOR_FIXES = "Minor Fixes and Nitpicks"
    TASK_COMPLETION = "Task Completion"
    CODE_IMPROVEMENTS = "Code Clarifications and Improvements"
    TECHNICAL_ISSUES = "Technical Issues and Questions"

# ClusterLabel class


class ClusterLabel:
    def __init__(self, cluster_num: int, cluster_name: ClusterName, qualitative_score: str):
        """
        Initializes a ClusterLabel instance with a qualitative score.

        Args:
            cluster_num (int): The cluster number.
            cluster_name (ClusterName): The label (Enum) associated with the cluster.
            qualitative_score (str): The qualitative score of the cluster (e.g., "low", "medium", "high").
        """
        self.cluster_num = cluster_num
        self.cluster_name = cluster_name
        self.qualitative_score = qualitative_score

    def __str__(self):
        """
        Returns a human-readable string representation of the ClusterLabel instance.
        """
        return f"Cluster {self.cluster_num}: {self.cluster_name.value} (Qualitative Score: {self.qualitative_score})"

    @staticmethod
    def get_label_from_cluster(cluster_num: int):
        """
        Maps a cluster number to a ClusterLabel instance with the appropriate label and qualitative score.

        Args:
            cluster_num (int): The cluster number.

        Returns:
            ClusterLabel: The corresponding label for the given cluster number.
        """
        cluster_mapping = {
            0: {"label": ClusterName.UI_UX_TESTING, "qualitative_score": "high"},
            1: {"label": ClusterName.CODE_REFACTORING, "qualitative_score": "medium"},
            2: {"label": ClusterName.POSITIVE_FEEDBACK, "qualitative_score": "medium"},
            3: {"label": ClusterName.MINOR_FIXES, "qualitative_score": "high"},
            4: {"label": ClusterName.TASK_COMPLETION, "qualitative_score": "low"},
            5: {"label": ClusterName.CODE_IMPROVEMENTS, "qualitative_score": "high"},
            6: {"label": ClusterName.TECHNICAL_ISSUES, "qualitative_score": "high"}
        }

        # Get the cluster name and qualitative score from the mapping
        cluster_info = cluster_mapping.get(cluster_num)
        if cluster_info is None:
            raise ValueError(f"Invalid cluster number: {cluster_num}")

        # Return a ClusterLabel instance
        return ClusterLabel(cluster_num, cluster_info['label'], cluster_info['qualitative_score'])


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

    def add_cluster_details_to_embeddings(self, embeddings: List[Dict[str, Any]], labels: List[int]):
        """
        Adds cluster details (cluster number, name, and qualitative score) to each embedding dictionary.

        Args:
            embeddings (List[Dict[str, Any]]): The list of embeddings with metadata.
            labels (List[int]): The cluster labels returned from the clustering method.
        """
        num_embeddings: int = len(embeddings)
        for i, embedding_data in enumerate(embeddings):
            print(f"\r\tAdding cluster details to embedding {
                  i + 1} of {num_embeddings}", end="")
            cluster_num = labels[i]
            # Get the cluster label instance (with the name and qualitative score)
            cluster_label = ClusterLabel.get_label_from_cluster(cluster_num)

            # Add cluster details to the embedding data
            embedding_data['cluster_num'] = cluster_label.cluster_num
            embedding_data['cluster_name'] = cluster_label.cluster_name.value
            embedding_data['qualitative_score'] = cluster_label.qualitative_score
        print("\n.")

        return embeddings

    def test_cluster_with_llm(self, cluster_num: int, list_texts: List[str], openai_client: ElOpenAI):
        """
        Test a given cluster by asking the LLM (via OpenAI API) to check the cluster name and provide a recommendation.

        Args:
            cluster_num (int): The number of the cluster to test.
            list_texts (List[str]): A list of text strings representing the contents of the cluster.
            openai_client (ElOpenAI): The OpenAI client to send the prompt to.

        Returns:
            ClusterNameStructuredOutput: The structured response from the LLM containing the recommended cluster name.
        """
        # Get the cluster name from the static mapping
        cluster_label = ClusterLabel.get_label_from_cluster(cluster_num)

        # Call the OpenAI client to test the cluster with LLM
        structured_output = openai_client.test_cluster_with_llm(
            cluster_name_to_check=cluster_label.cluster_name.value,
            list_texts=list_texts
        )

        # Return the structured output
        return structured_output

    def test_clusters(self, embeddings: List[Dict[str, Any]], kmeans_labels: List[int], openai_client: ElOpenAI):
        """
        Test clusters by sampling 5 comments from each cluster and calling test_cluster_with_llm.

        Args:
            embeddings (List[Dict[str, Any]]): List of embeddings with metadata.
            kmeans_labels (List[int]): List of cluster labels.
            openai_client (ElOpenAI): The OpenAI client to use for sending the LLM prompt.
        """
        # Step 1: Convert the embeddings to a Pandas DataFrame
        df = pd.DataFrame(embeddings)

        # Add the K-Means labels to the DataFrame
        df['cluster_num'] = kmeans_labels

        # Step 2: Loop through each cluster and sample 5 comments
        unique_clusters = df['cluster_num'].unique()

        for cluster_num in unique_clusters:
            # Sample 5 comments from this cluster
            cluster_sample = df[df['cluster_num'] == cluster_num].sample(5)

            # Convert the sample to a list of text strings
            list_texts = cluster_sample['text'].tolist()

            # Call test_cluster_with_llm for this cluster
            result = self.test_cluster_with_llm(
                cluster_num, list_texts, openai_client)

            # Output the result for inspection (you can store it later if needed)
            print(f"Cluster {cluster_num}: Recommended Cluster Name: {
                  result.recommended_cluster_name}")

        print("Cluster testing with LLM complete.")

    def name_clusters_with_llm(self, embeddings: List[Dict[str, Any]], kmeans_labels: List[int], openai_client: ElOpenAI):
        """
        Name clusters by sampling comments from each cluster and asking the LLM to suggest a name.

        Args:
            embeddings (List[Dict[str, Any]]): List of embeddings with metadata.
            kmeans_labels (List[int]): List of cluster labels.
            openai_client (ElOpenAI): The OpenAI client to use for querying the LLM.

        Returns:
            List[Dict[str, Any]]: The embeddings updated with LLM-suggested cluster names.
        """
        # Step 1: Convert embeddings to a Pandas DataFrame
        df = pd.DataFrame(embeddings)

        # Add the cluster numbers to the DataFrame
        df['cluster_num'] = kmeans_labels

        # Initialize a dictionary to store cluster names suggested by the LLM
        cluster_name_mapping = {}

        # Step 2: Loop through each cluster and query the LLM for a cluster name
        unique_clusters = df['cluster_num'].unique()
        for cluster_num in unique_clusters:
            # Sample 5 comments from the cluster
            cluster_sample = df[df['cluster_num'] == cluster_num].sample(5)
            list_texts = cluster_sample['text'].tolist()

            # Query the LLM to suggest a name for the cluster
            result = openai_client.test_cluster_with_llm(
                cluster_name_to_check=f"Cluster {cluster_num}",
                list_texts=list_texts
            )

            # Store the suggested name for the cluster
            cluster_name_mapping[cluster_num] = result.recommended_cluster_name
            print(f"Cluster {cluster_num}: Suggested Name: {
                  result.recommended_cluster_name}")

        # Step 3: Apply the LLM-suggested cluster names to the embeddings
        df['cluster_name'] = df['cluster_num'].map(cluster_name_mapping)

        # Return the updated list of embeddings with cluster names
        return df.to_dict(orient='records')


if __name__ == "__main__":
    filename = "review_embeddings.csv"

    embeddings = []

    # Check if the CSV file exists
    if os.path.exists(filename):
        print(f"Loading embeddings from {filename}...")

        try:
            # Use Pandas to read the CSV file
            df = pd.read_csv(filename)

            # Convert 'embedding' string back to a list of floats (if necessary)
            df['embedding'] = df['embedding'].apply(lambda emb: eval(emb))

            # Convert the DataFrame back to a list of dictionaries (if needed)
            embeddings = df.to_dict(orient='records')

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
            embeddings = embedding_label.generate_review_embeddings(limit=250)

            if len(embeddings) == 0:
                raise ValueError("No embeddings were generated.")

            print(f"New embeddings generated.")

        except Exception as e:
            print(f"An error occurred while generating embeddings: {e}")
            exit(1)

    # Initialize the ClusteringManager with embeddings
    manager = ClusteringManager(embeddings)

    # Step 1: Reduce dimensionality to 50 dimensions for clustering
    reduced_embeddings = manager.reduce_with_pca(n_components=50)

    # Step 2: Run K-Means clustering and get the labels (array of integers)
    kmeans_labels = manager.kmeans_clustering(reduced_embeddings, n_clusters=7)

    # Step 3: Further reduce dimensionality to 2 dimensions for plotting
    # reduced_embeddings_2d = manager.reduce_with_pca(n_components=2)

    # Step 4: Plot the K-Means clustering result
    # manager.plot_clusters(reduced_embeddings_2d,
    #                       kmeans_labels, "K-Means Clustering")

    # Initialize the OpenAI client
    openai_client = ElOpenAI()

    # Step 3: Name the clusters using LLM
    embeddings_with_clusters = manager.name_clusters_with_llm(
        embeddings, kmeans_labels, openai_client)

    # Step 4: Convert the embeddings_with_clusters to a Pandas DataFrame
    df = pd.DataFrame(embeddings_with_clusters)

    # Step 5: Drop the 'user' column if it exists
    df = df.drop(columns=['user'], errors='ignore')

    # Step 6: Convert 'embedding' list to string for CSV saving
    df['embedding'] = df['embedding'].apply(lambda emb: str(emb))

    # Step 7: Save DataFrame to CSV
    df.to_csv(filename, index=False)

    print(f"CSV successfully saved to {filename}.")
