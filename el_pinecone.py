import json
import os
from pinecone import Pinecone, ServerlessSpec, UpsertResponse, UnauthorizedException
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple, Union
from el_openai import ElOpenAI
import pandas as pd
import re
import numpy as np

from el_snowflake import ElSnowflake


class ElPinecone:
    """
    Class for interacting with Pinecone to store and query embeddings.
    Sequence:   1 - Constructor: Initialize Pinecone client
                2 - Create or reuse existing index
                3 - Upsert vectors (from model embeddings)
                4 - Query vector (with a search embedding)

    Test mode:  If PINECONE_TEST_MODE is set to True, all indexes in the project are deleted.
    This is transparent to the client. We delete everything during ElPinecone initialization.
    """
    # Class constants for Pinecone pricing
    STORAGE_COST_PER_GB_PER_HOUR = 0.00045
    WRITE_COST_PER_MILLION_UNITS = 2.00
    READ_COST_PER_MILLION_UNITS = 8.25
    COST_SCALE_FACTOR = 1_000
    BYTES_PER_DIMENSION = 4
    VECTORS_PER_UPSERT_BATCH = 250
    MINIMUM_SCORE = 0.22    # Minimum score for filtering results
    LOW_QUALITY_MINIMUM_SCORE = 0.1    # Minimum score for filtering results

    def __init__(self, embedding_model_name="", base_index_name: str = "", metric: str = "", dimension: int = 0):
        load_dotenv()  # Load environment variables from .env file
        self.api_key: str = os.getenv("PINECONE_API_KEY")
        self.project_name: str = os.getenv(
            "PINECONE_PROJECT_NAME", "ghvector")
        # These call all be overriden with create_index
        if metric == "":
            self.metric: str = os.getenv("PINECONE_METRIC", "cosine")
        else:
            self.metric = metric
        if dimension == 0:
            self.dimension = int(os.getenv("EMBEDDING_DIMENSIONS", 1536))
        else:
            self.dimension = dimension
        if base_index_name == "":
            self.base_index_name: str = os.getenv(
                "PINECONE_BASE_INDEX_NAME", "el")
        else:
            self.base_index_name = base_index_name
        if embedding_model_name == "":
            self.embedding_model_name: str = ElOpenAI.embedding_model_default
        else:
            self.embedding_model_name = embedding_model_name

        self.pc = Pinecone(api_key=self.api_key)

        # Until these vars are set, we don't have a valid index and can't upsert, etc.
        # So we'll set them in the get_and_prep_index function
        self.index_name = ""
        self.index_description = None
        self.index_host = ""
        self.index = None

    def delete_all_indexes(self):
        """
        Deletes all indexes in the Pinecone project. DANGER!!
        """
        # Test mode: delete all indexes in the project
        if os.getenv("PINECONE_TEST_MODE", "false").lower() == "true":
            self._delete_all_indexes()

    def get_and_prep_index(self) -> str:
        """
        Creates or reuses a Pinecone index based on the specified embedding model and dimensions.
        """
        self.index_name = self._create_or_reuse_index(
            self.embedding_model_name, self.dimension)
        self.index_description: str = self.pc.describe_index(self.index_name)
        self.index_host = self.index_description.host
        self.index = self.pc.Index(self.index_name, host=self.index_host)
        return self.index_name

    def _check_index_exists(self, index_name: str) -> bool:
        """
        Checks if a Pinecone index with the given name already exists.

        Args:
            index_name (str): The name of the index to check.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        return index_name in self.pc.list_indexes().names()

    def _create_or_reuse_index(self, embedding_model_name: str, dimension: int) -> str:
        """
        Creates a new Pinecone index or reuses an existing one if it already exists.

        Args:
            model_name (str): The name of the embedding model being used, included in the index name.
            dimensions (int): The number of dimensions for the embedding vectors.
        """
        # Construct and clean the initial index name
        index_name = f"{self.base_index_name}_{
            embedding_model_name[:20]}_{dimension}"
        cleaned_index_name = re.sub(r'[^a-z0-9\-]', '-', index_name.lower())
        self.index_name = cleaned_index_name[:45]

        # Check if the index already exists using the check_index_exists function
        if self._check_index_exists(self.index_name):
            pass
            # print(
            #     f"\tIndex '{self.index_name}' already exists. Reusing the existing index.")
        else:
            # Create the new index if it doesn't exist
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    region=os.getenv("PINECONE_REGION", "us-east-1")
                )
            )
            print(f"\tCreated Pinecone index '{self.index_name}' with {
                  dimension} dimensions and '{self.metric}' metric.")
        return self.index_name

    def _connect_to_index(self):
        """
        Connects to the Pinecone index or creates it if it doesn't exist.
        """
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    region=os.getenv("PINECONE_REGION", "us-east-1")
                )
            )
        self.index = self.pc.Index(self.index_name)
        print(f"Connected to Pinecone index: {self.index_name}")

    def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> int:
        """
        Upserts a batch of vectors into the Pinecone index.

        Args:
            vectors (List[Dict[str, Any]]): A list of dictionaries where each dictionary contains:
                - 'id': The unique ID for the vector.
                - 'values': The vector embedding.
                - 'metadata': Optional metadata associated with the vector.
        """
        if not self.index:
            self.get_and_prep_index()

        total_upserted_count: int = 0
        num_vectors = len(vectors)

        for i in range(0, num_vectors, ElPinecone.VECTORS_PER_UPSERT_BATCH):
            chunk = vectors[i:i + ElPinecone.VECTORS_PER_UPSERT_BATCH]
            print(f"\t\rUpserting vectors {
                  i + 1} to {i + ElPinecone.VECTORS_PER_UPSERT_BATCH} of {num_vectors}")
            upsert_response = self.index.upsert(chunk)
            upserted_count = upsert_response.get("upserted_count", 0)
            total_upserted_count += upserted_count

        print("\n")
        if total_upserted_count != num_vectors:
            print(f"\tUpserted {
                  total_upserted_count}; Num vectors attempted: {num_vectors}")

        return total_upserted_count

    def filter_results_by_score(self, results: List[Dict[str, Any]], min_score: float) -> List[Dict[str, Any]]:
        """
        Filters the results based on the minimum score.

        Args:
            results (List[Dict[str, Any]]): The list of results to filter.
            min_score (float): The minimum score to filter the results.

        Returns:
            List[Dict[str, Any]]: The filtered list of results.
        """
        return [result for result in results if result.get("score", 0.0) >= min_score]

    def query_vector(self, vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Queries the Pinecone index with a vector and returns the top_k most similar vectors.

        Args:
            vector (List[float]): The vector to similarity match on query.
            top_k (int): The number of top results to return.

        Returns:
            List[Dict[str, Any]]: A list of the top_k most similar vectors.
        """
        if not self.index:
            self.get_and_prep_index()
        try:
            # Perform the query
            result = self.index.query(
                vector=vector, top_k=top_k, includeMetadata=True)
            # print(f"Query result: {result['matches']}")
            return result['matches']  # Return the list of matches

        except UnauthorizedException as e:
            # Handle unauthorized access specifically
            print(f"UnauthorizedException: {e}")
            print(
                "Please check your API key and ensure you have the necessary permissions.")
            raise  # Re-raise the exception to signal the failure

        except Exception as e:
            # General exception handling
            print(f"An error occurred during the query: {e}")
            raise  # Re-raise the exception to ensure it's not silently ignored

    def delete_vector(self, vector_id: str):
        """
        Deletes a vector from the Pinecone index by its ID.

        Args:
            vector_id (str): The ID of the vector to delete.
        """
        if not self.index:
            self.get_and_prep_index()
        self.index.delete(ids=[vector_id])
        print(f"Deleted vector with ID: {
              vector_id} from Pinecone index: {self.index_name}")

    def fetch_vector(self, vector_id: str) -> Dict[str, Any]:
        """
        Fetches a vector from the Pinecone index by its ID.

        Args:
            vector_id (str): The ID of the vector to fetch.

        Returns:
            Dict[str, Any]: The fetched vector data.
        """
        if not self.index:
            self.get_and_prep_index()
        result = self.index.fetch(ids=[vector_id])  # Fetch the vector by ID
        return result  # Return the fetched vector data

    def describe_index(self):
        """
        Describe the index to get stats and other metadata.

        Returns:
            Dict[str, Any]: The stats of the Pinecone index.
        """
        if not self.index:
            self.get_and_prep_index()
        return self.index.describe_index_stats()  # Get and return the index stats

    def _delete_all_indexes(self):
        """
        Deletes all indexes in the Pinecone project. DANGER!!
        """
        # List all indexes in the project
        indexes = self.pc.list_indexes().names()

        if not indexes:
            print("No indexes found in the project.")
            return

        # Iterate through the list of indexes and delete each one
        count: int = 0
        for index_name in indexes:
            print(f"\r\tDeleting index: {index_name}", end="")
            self.pc.delete_index(name=index_name)
            count += 1

        print(f"\n\t{count} indexes have been deleted.")

    def calculate_storage_cost(self, dimensions: int, num_vectors: int = 1, hours: int = 1) -> float:
        """
        Calculates the cost of storing 1000 vectors in Pinecone.

        Args:
            num_vectors (int): The number of vectors being stored.
            hours (int): The number of hours the vectors are stored (default is 1 hour).

        Returns:
            float: The estimated cost of storing the vectors for the specified time.
        """
        # Convert dimensions to bytes
        vector_size_bytes = dimensions * self.BYTES_PER_DIMENSION
        bytes_per_gb = 1024 * 1024 * 1024

        # Calculate total storage in GB
        total_storage_gb = (num_vectors * vector_size_bytes) / bytes_per_gb

        # Total storage cost
        return total_storage_gb * self.STORAGE_COST_PER_GB_PER_HOUR * hours * self.COST_SCALE_FACTOR

    def calculate_write_cost(self, num_vectors: int = 1) -> float:
        """
        Calculates the cost of writing 1000 vectors to Pinecone.

        Args:
            num_vectors (int): The number of vectors being written.

        Returns:
            float: The estimated cost of writing the vectors.
        """
        # Pinecone charges $2.00 per 1M Write Units
        write_units = num_vectors / 1_000_000
        return write_units * self.WRITE_COST_PER_MILLION_UNITS * self.COST_SCALE_FACTOR

    def calculate_read_cost(self, num_queries: int = 1) -> float:
        """
        Calculates the cost of querying 1000 vectors in Pinecone.

        Args:
            num_queries (int): The number of queries made.

        Returns:
            float: The estimated cost of querying the vectors.
        """
        # Pinecone charges $8.25 per 1M Read Units
        read_units = num_queries / 1_000_000
        return read_units * self.READ_COST_PER_MILLION_UNITS * self.COST_SCALE_FACTOR


def gen_embedding_and_upsert(el_pc: ElPinecone, dict_test: Dict, el_openai: ElOpenAI) -> int:
    """
    Stores a test vector from the test dictionary.

    Args:
        el_openai (OpenAI): The OpenAI instance used to generate embeddings.
    """

    # Generate embedding for the dummy function using OpenAI
    function_response = el_openai.generate_embeddings(dict_test["text"])

    # Access the embedding vector directly from the response
    function_embedding = function_response['embedding']

    # Upsert the embedding to Pinecone
    upserted_count: int = el_pc.upsert_vectors([{
        "id": dict_test["id"],
        "values": function_embedding,
        "metadata": dict_test["metadata"]
    }])
    return upserted_count


def test_embedding_search(el_pc: ElPinecone, dict_test: Dict, el_openai: ElOpenAI) -> pd.DataFrame:
    """
    Tests the vector query by generating an embedding for a dummy function and querying with a related prompt.

    Args:
        el_openai (ElOpenAI): The ElOpenAI instance used to generate embeddings.
    """

    query_response = el_openai.generate_embeddings(dict_test["prompt"])

    # Access the embedding vector directly from the response
    query_embedding = query_response['embedding']

    # Query Pinecone using the query embedding
    print(f"\tQuerying Pinecone {el_pc.index_name} with the prompt: '{
        dict_test['prompt']}'")

    results = el_pc.query_vector(vector=query_embedding, top_k=5)

    # Calculate the cost of generating embeddings based on the number of tokens
    num_tokens: int = el_openai.count_tokens(
        dict_test["text"]) + el_openai.count_tokens(dict_test["prompt"])
    openai_cost: float = el_openai.calculate_cost(num_tokens)
    # Structure results for DataFrame compatibility
    structured_results = []
    for result in results:
        row = {
            "embedding_model": el_openai.embedding_model,
            "dimensions": el_openai.dimensions,
            "index_name": el_pc.index_name,
            "prompt": dict_test["prompt"],
            "text": dict_test["text"],
            "result_id": result.get("id", ""),
            "score": float(result.get("score", 0.0)),
            "num_tokens": num_tokens,
            "cost": openai_cost,
            "vectordb_storage_cost": el_pc.calculate_write_cost() + el_pc.calculate_storage_cost(el_openai.dimensions),
            "vectordb_read_cost": el_pc.calculate_read_cost(),
        }
        structured_results.append(row)

    # Convert structured results into a DataFrame
    results_df = pd.DataFrame(structured_results)

    return results_df


if __name__ == "__main__":
    el_pinecone = ElPinecone()
