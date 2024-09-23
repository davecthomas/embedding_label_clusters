
from typing import List
import pandas as pd
from el_openai import ElOpenAI
from el_snowflake import ElSnowflake


class EmbeddingLabel:
    def __init__(self):
        self.openai_client = ElOpenAI()
        self.snowflake_client = ElSnowflake()

    def generate_review_embeddings(self, limit: int, batch_size: int = 50) -> pd.DataFrame:
        """
        Fetches review comments from Snowflake, passes them to the ElOpenAI client to generate embeddings,
        and returns the updated DataFrame with an 'embedding' column.

        Args:
            limit (int): The maximum number of review comments to process.
            batch_size (int): The number of texts to process in each batch (default is 100).

        Returns:
            pd.DataFrame: A DataFrame with the original review comments and their embeddings.
        """
        # Fetch the review comments as a DataFrame from Snowflake
        df: pd.DataFrame = self.snowflake_client.get_review_comments(
            limit=limit)

        # Pass the DataFrame to ElOpenAI to generate embeddings
        df_with_embeddings = self.openai_client.generate_embeddings(
            df, batch_size=batch_size)

        # Return the updated DataFrame with embeddings
        return df_with_embeddings


if __name__ == "__main__":
    # Initialize the EmbeddingLabel client
    embedding_label = EmbeddingLabel()
