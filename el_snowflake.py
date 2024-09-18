import json
import os
import random
import time
from typing import Dict, Optional, List, Any, Tuple
import numpy as np
import pandas as pd
import snowflake.connector
from datetime import datetime, timezone

from el_openai import ElOpenAI


class ElSnowflake:

    def __init__(self):
        self.dict_db_env = None
        self.conn: Optional[snowflake.connector.SnowflakeConnection] = None
        self.get_db_env()
        self.backoff_delays = [1, 2, 4, 8, 16]  # Delays in seconds

    def __del__(self):
        """Destructor to ensure the Snowflake connection is closed."""
        try:
            self.close_connection()
        except Exception as e:
            print(f"Error closing Snowflake connection: {e}")

    def close_connection(self):
        """Closes the Snowflake connection if it's open."""
        if self.conn is not None and not self.conn.is_closed():
            try:
                self.conn.close()
                print("Snowflake connection closed.")
            except snowflake.connector.Error as e:
                print(f"Error closing Snowflake connection: {e}")
                raise
            finally:
                self.conn = None  # Reset the cached connection to None

    def get_db_env(self) -> Dict[str, str]:
        """Fetches database environment variables."""
        if self.dict_db_env is None:
            self.dict_db_env = {
                "snowflake_user": os.getenv("SNOWFLAKE_USER"),
                "snowflake_role": os.getenv("SNOWFLAKE_ROLE"),
                "snowflake_password": os.getenv("SNOWFLAKE_PASSWORD"),
                "snowflake_account": os.getenv("SNOWFLAKE_ACCOUNT"),
                "snowflake_warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
                "snowflake_db": os.getenv("SNOWFLAKE_DB"),
                "snowflake_schema": os.getenv("SNOWFLAKE_SCHEMA")
            }
        return self.dict_db_env

    def get_snowflake_connection(self) -> snowflake.connector.SnowflakeConnection:
        """Establishes a connection to Snowflake with hardcoded backoff delay.
         Returns:
            snowflake.connector.SnowflakeConnection: The Snowflake connection object.
        """
        if self.conn is None or self.conn.is_closed():
            dict_db_env = self.get_db_env()
            for attempt, delay in enumerate(self.backoff_delays, 1):
                try:
                    db = dict_db_env["snowflake_db"]
                    schema = dict_db_env["snowflake_schema"]
                    self.conn = snowflake.connector.connect(
                        user=dict_db_env["snowflake_user"],
                        password=dict_db_env["snowflake_password"],
                        account=dict_db_env["snowflake_account"],
                        warehouse=dict_db_env["snowflake_warehouse"],
                        database=db,
                        schema=schema,
                        timeout=30,  # Set a timeout for connection
                        role=dict_db_env.get("snowflake_role"),
                    )
                    break
                except snowflake.connector.errors.OperationalError as e:
                    print(f"Connection attempt {attempt} failed: {
                          e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
            if self.conn is None or self.conn.is_closed():
                raise Exception(
                    "Could not connect to Snowflake after multiple attempts.")
        return self.conn

    def test_connection(self) -> bool:
        """Tests opening a connection to Snowflake and performing a basic SELECT query."""
        try:
            conn = self.get_snowflake_connection()
            query = "SELECT CURRENT_TIMESTAMP;"
            df = pd.read_sql(query, conn)

            print("Snowflake connection test successful. Retrieved rows:")
            print(df)
            return True
        except Exception as e:
            print(f"Snowflake connection test failed with error: {e}")
            return False

    def get_review_comments(self, limit: int = 100) -> pd.DataFrame:
        """
        Fetches review comments from Snowflake along with the comment_id.
        """
        if limit:
            limit = f"LIMIT {limit}"
        try:
            conn = self.get_snowflake_connection()
            query = f"""
                SELECT
                    "comment_id",   -- Add comment_id to fetch
                    "repo_name",
                    "pr_number",
                    "user_login",
                    "body",
                    "created_at"
                FROM
                    "pr_review_comments"
                WHERE
                    "body" IS NOT NULL
                {limit};
            """
            df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            print(f"Error fetching pr_review_comments: {e}")
            return None

    def store_classification(self, df: pd.DataFrame):
        """
        Stores classification (cluster name), comment body, and quality score in the training table based on comment_id.

        Args:
            df (pd.DataFrame): DataFrame containing comment_id, cluster names (label), comment body, and quality score.
        """
        try:
            conn = self.get_snowflake_connection()

            # Prepare the INSERT or UPDATE query to store the classifications
            for i, row in df.iterrows():
                print(f"\r\tStoring classification {i+1}", end="")
                comment_id = row['comment_id']
                label = row['cluster_name']  # This is the cluster name (label)
                # Include the comment body for human review of classifications
                body = row['body']
                quality_score = row['quality_score']

                try:
                    # Merge query for Snowflake with correct SQL syntax
                    query = f"""
                         MERGE INTO "pr_review_comments_training" AS target
                         USING (
                             SELECT
                                 '{comment_id}' AS "comment_id",
                                 '{label.replace("'", "''")}' AS "label",
                                 '{body.replace("'", "''")}' AS "body",
                                 {quality_score} AS "quality_score"
                         ) AS source
                         ON target."comment_id" = source."comment_id"
                         WHEN MATCHED THEN
                             UPDATE SET target."label" = source."label", target."quality_score" = source."quality_score"
                         WHEN NOT MATCHED THEN
                             INSERT ("comment_id", "label",
                                     "body", "quality_score")
                             VALUES (source."comment_id", source."label",
                                     source."body", source."quality_score");
                     """
                    # Execute the query
                    with conn.cursor() as cur:
                        cur.execute(query)
                        # to debug this, output query.replace("\\", "")
                        # print(f"Classification for comment_id {comment_id} saved.")
                except Exception as e:
                    print(f"\nError saving classification for comment_id {
                          comment_id}: {e}")

        except Exception as e:
            print(f"\nError storing classifications: {e}")

    def store_classification_batch(self, df: pd.DataFrame, batch_size: int = 100):
        """
        Stores classification (cluster name), comment body, and quality score in the training table based on comment_id.
        This version batches the data into a single query to speed up the process.

        Args:
            df (pd.DataFrame): DataFrame containing comment_id, cluster names (label), comment body, and quality score.
            batch_size (int): Number of rows to include in each batch. Default is 100.
        """
        try:
            conn = self.get_snowflake_connection()

            # Process the DataFrame in batches
            for start in range(0, len(df), batch_size):
                print(f"\r\tStoring batch {
                      start // batch_size + 1} (rows {start} to {start + len(batch_df) - 1})", end="")
                batch_df = df.iloc[start:start + batch_size]

                # Create a list of values to be inserted/updated in SQL
                values_list = []
                for i, row in batch_df.iterrows():
                    comment_id = row['comment_id']
                    label = row['cluster_name']
                    # Handle single quotes in SQL
                    body = row['body'].replace("'", "''")
                    quality_score = row['quality_score']

                    # Add to list (SQL friendly format)
                    values_list.append(
                        f"('{comment_id}', '{label.replace("'", "''")}', '{body}', {quality_score})")

                # Join all rows into a single query for batch insert/merge
                values_string = ", ".join(values_list)

                # Create the batch MERGE SQL query
                query = f"""
                    MERGE INTO "pr_review_comments_training_test_batch" AS target
                    USING (VALUES {values_string}) AS source("comment_id", "label", "body", "quality_score")
                    ON target."comment_id" = source."comment_id"
                    WHEN MATCHED THEN
                        UPDATE SET target."label" = source."label", target."quality_score" = source."quality_score"
                    WHEN NOT MATCHED THEN
                        INSERT ("comment_id", "label", "body", "quality_score")
                        VALUES (source."comment_id", source."label", source."body", source."quality_score");
                """

                # Execute the query
                with conn.cursor() as cur:
                    cur.execute(query)
                print(f"\rBatch {start // batch_size + 1} completed.", end="")

        except Exception as e:
            print(f"\nError storing classifications: {e}")


if __name__ == "__main__":
    # Initialize the Snowflake client
    snowflake_client = ElSnowflake()

    # # Test the connection to Snowflake and perform a basic SELECT query
    # connection_test_result = snowflake_client.test_connection()

    # # Output the result of the connection test
    # print(f"Connection Test Passed: {connection_test_result}")

    # Fetch review comments from Snowflake
    df_review_comments: pd.DataFrame = snowflake_client.get_review_comments()
    print(df_review_comments.head())
