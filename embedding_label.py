
from typing import List
import pandas as pd
from el_openai import ElOpenAI
from el_snowflake import ElSnowflake


class EmbeddingLabel:
    def __init__(self):
        self.openai_client = ElOpenAI(dimensions=ElOpenAI.DIMENSIONS_MIN)
        self.snowflake_client = ElSnowflake()

    def generate_review_embeddings(self, limit: int = 100) -> List[List[float]]:
        """
        Generates embeddings for review comments and stores them in a dataframe.
        Returns:
            List[List[float]]: A list of embeddings for review comments.
        """
        list_of_embeddings: List[List[float]] = []
        df: pd.DataFrame = self.snowflake_client.get_review_comments(
            limit=limit)
        # loop through dataframe and generate_embeddings
        for index, row in df.iterrows():
            text = row["body"]
            embedding = self.openai_client.generate_embedding(text)
            if embedding:
                row["embedding"] = embedding
                list_of_embeddings.append(embedding)
        return list_of_embeddings


if __name__ == "__main__":
    # Initialize the EmbeddingLabel client
    embedding_label = EmbeddingLabel()
