# el_openai.py

import os
import random
import time
import httpx
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any
from openai.types import EmbeddingCreateParams, CreateEmbeddingResponse
import pandas as pd
import tiktoken
import csv
import ast  # Import the ast module to safely evaluate string representations of lists
from pydantic import BaseModel, field_validator


class ClusterNameStructuredOutput(BaseModel):
    cluster_number: int  # The cluster number we are naming
    suggested_cluster_name: str  # The name suggested by the LLM
    quality_score: int = None  # The LLM-evaluated quality score (1-5)

    @field_validator('cluster_number', 'suggested_cluster_name', 'quality_score', mode='before')
    def validate_fields(cls, value, info):
        """
        Validate that 'cluster_number', 'suggested_cluster_name', and 'quality_score' are properly set.
        """
        if info.field_name == 'cluster_number':
            if not isinstance(value, int) or value < 0:
                raise ValueError("Invalid or missing 'cluster_number'")
        elif info.field_name == 'suggested_cluster_name':
            if not isinstance(value, str) or value == "":
                raise ValueError("Invalid or missing 'suggested_cluster_name'")
        elif info.field_name == 'quality_score':
            if not isinstance(value, int) or not (1 <= value <= 5):
                # Enforce fallback score if value is out of range
                print(f"Warning: Invalid quality score '{
                      value}' found. Removing this value.")
                return None
        return value


class ElOpenAI:
    DIMENSIONS_GEN3_OPENAI = 1536

    # Dictionary containing LLM token limits (completions models)
    LLM_TOKEN_LIMITS = {
        "gpt-3.5-turbo": 4096,
        "text-davinci-003": 4096,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4o": 4096,
        "gpt-4o-mini": 2048
    }

    # Static list of embedding models with their settings
    embedding_models: Dict = {
        "text-embedding-ada-002": {"dimensions": 1536, "pricing_per_token": 0.0004},
        "text-embedding-3-small": {"dimensions": 1536, "pricing_per_token": 0.00025},
        "text-embedding-3-large": {"dimensions": 3072, "pricing_per_token": 0.0005}}

    embedding_model_default = "text-embedding-3-small"

    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = DIMENSIONS_GEN3_OPENAI):
        """
        Initializes the ElOpenAI class, setting the model and dimensions.

        Args:
            model (str): The embedding model to use.
            dimensions (int): The number of dimensions for the embeddings.
        """
        load_dotenv()  # Load environment variables from .env file
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model = model  # Use the model passed to the constructor
        self.dimensions = dimensions  # Use the dimensions passed to the constructor
        self.user = os.getenv("OPENAI_USER", "default_user")
        self.completions_model = os.getenv(
            "OPENAI_COMPLETIONS_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=self.api_key)
        # Exponential backoff delays in seconds
        self.backoff_delays = [1, 2, 4, 8, 16]
        # Since removing exclusion terms requires a more expensive API call, we default to False
        self.exclusion_prompt_support = os.getenv(
            "OPENAI_EXCLUSION_PROMPTS", "false").lower() == "true"

    def get_llm_token_limit(self) -> int:
        """
        Retrieves the token limit for the current completions model (LLM).

        Returns:
            int: The token limit for the specified LLM.
        """
        return self.LLM_TOKEN_LIMITS.get(self.completions_model, 4096)  # Default to 4096 if model not found

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text based on a generic tokenizer cl100k_base.

        Args:
            text (str): The text to be tokenized.

        Returns:
            int: The number of tokens in the text.
        """
        # Get the appropriate tokenizer for the model
        tokenizer = tiktoken.get_encoding("cl100k_base")

        # Tokenize the text and count the tokens
        tokens = tokenizer.encode(text)
        return len(tokens)

    def calculate_cost(self, num_tokens: int) -> float:
        """
        Calculate the cost of generating embeddings based on the number of tokens.

        Args:
            num_tokens (int): The number of tokens used.

        Returns:
            float: The calculated cost.
        """
        pricing_per_token: float = self.embedding_models[self.embedding_model].get(
            "pricing_per_token", 0.0)
        cost = pricing_per_token * num_tokens
        return cost

    @staticmethod
    def normalize_l2(x) -> np.ndarray:
        """L2 normalizes an embedding vector to retain important properties."""
        x = np.array(x)
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm

    def generate_embedding(self, text: str) -> Dict[str, Any]:
        """
        Generates embedding for a given text using OpenAI's embeddings API.

        Args:
            text (str): The text for which to generate embeddings.

        Returns:
            Dict[str, Any]: A dictionary containing the embeddings and metadata.
        """
        if not text:
            raise ValueError("Text is required for generating embeddings.")
        # Construct the parameters using EmbeddingCreateParams
        params: EmbeddingCreateParams = {
            "input": [text],  # Input text as a list of strings
            "model": self.embedding_model,  # Model to use for embedding
        }

        max_retries = len(self.backoff_delays)

        for attempt in range(max_retries):
            try:
                # Call the API to create the embedding
                response: CreateEmbeddingResponse = self.client.embeddings.create(
                    **params)

                # Extract the embedding from the response
                # Note: embedding dimensions are immutable per model, so there's no purpose for this parameter
                embedding = response.data[0].embedding
                return {
                    "embedding": embedding,
                    "text": text,
                    "dimensions": self.dimensions,  # Dimensions of the embedding
                    "user": self.user  # User identifier
                }
            except (httpx.TimeoutException) as e:
                wait_time = self.backoff_delays[attempt]
                print(f"\tOpenAI embeddings attempt {
                      attempt + 1} failed: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time + random.uniform(0, 1))

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limit error
                    wait_time = self.backoff_delays[attempt]
                    print(f"\tRate limit reached. Retrying in {
                          wait_time} seconds...")
                    time.sleep(wait_time + random.uniform(0, 1))
                else:
                    print(f"\tHTTP error occurred: {e}")
                    raise

            except Exception as e:
                print(f"\tAn unexpected error occurred: {e}")
                raise

        raise TimeoutError(
            "Failed to generate embeddings after multiple retries due to repeated timeouts.")

    def sendPrompt(self, prompt: str = "", system_prompt="You are a helpful software coding assistant.") -> str:
        """
        Sends a prompt to the latest version of the OpenAI API for chat and returns the completion result.

        Args:
            prompt (str): The prompt string to send.

        Returns:
            str: The completion result as a string.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.completions_model,
                messages=[
                    {"role": "system",
                        "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract the response from the completion
            completion = response.choices[0].message.content

            # If the content seems truncated, send a follow-up request or handle continuation
            while response.choices[0].finish_reason == 'length':
                response = self.client.chat.completions.create(
                    model=self.completions_model,
                    messages=[
                        {"role": "system", "content": "Continue."},
                    ]
                )
                completion += response.choices[0].message.content
            return completion

        except Exception as e:
            print(f"An error occurred while sending the prompt: {e}")
            raise

    def name_cluster(self, cluster_number: int, list_texts: List[str], existing_names: List[str]) -> ClusterNameStructuredOutput:
        """
        Uses OpenAI's chat completions API to suggest a name for a given cluster number based on the cluster's text comments.

        Args:
            cluster_number (int): The number of the cluster to name.
            list_texts (List[str]): A list of text strings representing the contents of the cluster.
            existing_names (List[str]): A list of existing cluster names to avoid repetition.

        Returns:
            ClusterNameStructuredOutput: A structured response with the suggested cluster name.
        """
        # System prompt asking the LLM to suggest a name and evaluate the quality of the cluster
        system_prompt = (
            "Your task is to name a cluster of code review comments based on recurring patterns or key topics. "
            "Avoid names similar to existing clusters and assign a quality score (1-5) based on the specificity and usefulness of the comments, "
            "where 1 is for acknowledgements and 5 is for impactful suggestions. "
            "Here are the existing cluster names: "
            f"{', '.join(existing_names)}. ")

        # User prompt providing the cluster number and associated comments for analysis
        prompt = (
            "Suggest a descriptive name for cluster "
            f"{cluster_number}, avoiding similarity with these existing cluster names: "
            f"{existing_names}. "
            f"Here is a list of code review comments from this cluster to base your decision on: "
            f"{list_texts}. "
        )

        # Prepare the input messages for the LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        try:
            # Use the chat completion API with structured output parsing using Pydantic
            completion = self.client.beta.chat.completions.parse(
                model=self.completions_model,
                messages=messages,
                response_format=ClusterNameStructuredOutput,
            )

            # Validate and return the structured response
            return ClusterNameStructuredOutput(**completion.choices[0].message.parsed.model_dump())

        except Exception as e:
            print(f"An error occurred: {e}")
            return ClusterNameStructuredOutput(cluster_number=cluster_number, suggested_cluster_name="Unknown")

    def test_openai_connectivity(self, list_text: List[str] = ["Hello, World!"]) -> bool:
        """
        Tests connectivity to the OpenAI API by generating embeddings for the provided text.

        Args:
            list_text (List[str]): A list of text strings to embed. Defaults to ["Hello, World!"].

        Returns:
            bool: True if the API request is successful, False otherwise.
        """
        try:
            for text in list_text:
                embedding_data = self.generate_embedding(text)
                print(f"Successfully generated embedding for: {text}")
                print(f"Embedding: {
                      embedding_data['embedding'][:5]}... [truncated]")

            return True

        except Exception as e:
            print(f"OpenAI API connectivity test failed with error: {e}")
            return False

    def save_to_csv(embeddings: List[Dict[str, Any]], filename: str):
        """
        Saves a list of embeddings to a CSV file.

        Args:
            embeddings (List[Dict[str, Any]]): A list of dictionaries, each containing 'embedding', 'text', and other metadata.
            filename (str): The name of the CSV file to save the embeddings to.
        """
        if not embeddings or not isinstance(embeddings, list):
            raise ValueError(
                "Embeddings should be a non-empty list of dictionaries.")

        # Dynamically generate the header from the keys of the first dictionary
        header = embeddings[0].keys()

        try:
            with open(filename, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=header)

                # Write the header
                writer.writeheader()

                # Write each embedding to the CSV
                for embedding_data in embeddings:
                    # Convert the embedding vector into a string (e.g., "[1.23, 4.56, ...]")
                    embedding_data['embedding'] = str(
                        embedding_data['embedding'])
                    writer.writerow(embedding_data)

            print(f"Embeddings successfully saved to {filename}")

        except IOError as e:
            print(f"An error occurred while saving embeddings to CSV: {e}")

    @staticmethod
    def convert_embeddings_to_numeric_array(df: pd.DataFrame) -> np.ndarray:
        """
        Converts a DataFrame containing dictionaries of OpenAI embeddings (with metadata) 
        into a NumPy array of embeddings.

        Args:
            df (pd.DataFrame): DataFrame where each row contains an 'embedding' column,
                            which is a dictionary with metadata.

        Returns:
            np.ndarray: A 2D NumPy array where each row is a list of floats representing an embedding.
        """
        if 'embedding' not in df.columns:
            raise KeyError("'embedding' column is missing from the DataFrame.")

        # Initialize an empty list to store the embeddings
        embedding_list = []

        for index, row in df.iterrows():
            print(f"\r\tConverting embedding to numeric array for clustering {
                  index}", end="")

            # Get the actual embedding (list of floats)
            embedding_value = row['embedding']

            if embedding_value is not None:
                # Convert to NumPy array
                embedding_list.append(np.array(embedding_value, dtype=float))
            else:
                print(f"\nWarning: Skipping row {
                      index} due to missing embedding.")

        print("\n.")

        # Convert the list of embeddings to a 2D NumPy array
        return np.array(embedding_list)


if __name__ == "__main__":
    openai_client = ElOpenAI()
    # Test connectivity to the OpenAI API
    connectivity_test_result = openai_client.test_openai_connectivity()

    # Output the result of the connectivity test
    print(f"Connectivity Test Passed: {connectivity_test_result}")
