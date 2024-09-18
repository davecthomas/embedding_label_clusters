
from typing import List
import pandas as pd
from el_openai import ElOpenAI
from el_snowflake import ElSnowflake


class EmbeddingLabel:
    def __init__(self):
        self.openai_client = ElOpenAI()
        self.snowflake_client = ElSnowflake()

    def generate_review_embeddings(self, limit) -> pd.DataFrame:
        """
        Generates embeddings for review comments and stores them in a DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with the original review comments and their embeddings.
        """
        # Fetch the review comments as a DataFrame from Snowflake
        df: pd.DataFrame = self.snowflake_client.get_review_comments(
            limit=limit)

        # Loop through the DataFrame and generate embeddings for each comment
        num_rows: int = len(df)
        embeddings: List[List[float]] = []
        for index, row in df.iterrows():
            print(f"\r\tGenerating embeddings from review {
                  index + 1} of {num_rows}", end="")
            text = row["body"]
            embedding = self.openai_client.generate_embedding(text)
            if embedding:
                # Only extract the 'embedding' key (list of floats) and append to the list
                embeddings.append(embedding.get("embedding", None))

        # Add the embeddings to the DataFrame as a new column
        df['embedding'] = embeddings
        print("\n.")

        return df


if __name__ == "__main__":
    # Initialize the EmbeddingLabel client
    embedding_label = EmbeddingLabel()
