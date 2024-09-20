
-- PR review comments stored to do AI-based analysis of them
CREATE TABLE IF NOT EXISTS "pr_review_comments" (
    "comment_id" BIGINT PRIMARY KEY,
    "repo_name" VARCHAR(256),
    "pr_number" VARCHAR(64),
    "user_login" VARCHAR(256),
    "body" TEXT,
    "created_at" TIMESTAMP_NTZ
);

-- Staging table for PR review comments so we can more efficiently merge
CREATE TABLE IF NOT EXISTS "pr_review_comments_staging" (
    "comment_id" BIGINT PRIMARY KEY,
    "repo_name" VARCHAR(256),
    "pr_number" VARCHAR(64),
    "user_login" VARCHAR(256),
    "body" TEXT,
    "created_at" TIMESTAMP_NTZ
);

CREATE TABLE IF NOT EXISTS "pr_review_comments_training" (
    "comment_id" BIGINT PRIMARY KEY,
    "repo_name" VARCHAR(256),
    "pr_number" VARCHAR(64),
    "user_login" VARCHAR(256),
    "body" TEXT,
    "created_at" TIMESTAMP_NTZ,
    LABEL TEXT, -- this is our classification column. Due to a bug in Snowflake prediction, we can't use lower case here.
    "quality_score" INT DEFAULT NULL -- this is an optional column for quality score
    "embedding" ARRAY;    -- this is the embedding of the comment
);

CREATE TABLE IF NOT EXISTS "pr_review_comments_training_backup" (
    "comment_id" BIGINT PRIMARY KEY,
    "repo_name" VARCHAR(256),
    "pr_number" VARCHAR(64),
    "user_login" VARCHAR(256),
    "body" TEXT,
    "created_at" TIMESTAMP_NTZ,
    LABEL TEXT,  -- Copy of the original LABEL column
    "quality_score" INT DEFAULT NULL -- Copy of the original quality score column
);

