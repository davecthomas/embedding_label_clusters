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
import tiktoken
import csv


class ElOpenAI:
    DIMENSIONS_MIN = 256
    DIMENSIONS_DEFAULT = 512
    DIMENSIONS_GEN3_OPENAI = 1536

    # Static list of embedding models with their settings
    embedding_models: Dict = {
        "text-embedding-ada-002": {"dimensions": 1536, "pricing_per_token": 0.0004},
        "text-embedding-3-small": {"dimensions": 1536, "pricing_per_token": 0.00025},
        "text-embedding-3-large": {"dimensions": 3072, "pricing_per_token": 0.0005}}

    embedding_model_default = "text-embedding-3-small"

    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = 512):
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

    @staticmethod
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

        # Define the header based on the keys in the first dictionary
        header = ['text', 'embedding', 'dimensions', 'user']

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
    def convert_embeddings_to_numeric_array(embeddings: List[Dict[str, Any]]) -> np.ndarray:
        """
        Converts a list of dictionaries containing OpenAI embeddings into a NumPy array.

        Args:
            embeddings (List[Dict[str, Any]]): List of dictionaries where each dictionary contains an 'embedding' key.

        Returns:
            np.ndarray: A 2D NumPy array where each row is a list of floats representing an embedding.
        """
        if not isinstance(embeddings, list):
            raise TypeError(
                "Embeddings should be a list of dictionaries or a list of lists.")

        # Check if it's a list of dictionaries with the 'embedding' key
        if all(isinstance(e, dict) and 'embedding' in e for e in embeddings):
            return np.array([np.array(e['embedding'], dtype=float) for e in embeddings])

        # If it's already a list of lists, convert directly to NumPy array
        return np.array(embeddings, dtype=float)


if __name__ == "__main__":
    openai_client = ElOpenAI()
    # Test connectivity to the OpenAI API
    connectivity_test_result = openai_client.test_openai_connectivity()

    # Output the result of the connectivity test
    print(f"Connectivity Test Passed: {connectivity_test_result}")
