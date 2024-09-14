
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
    "label" VARCHAR(50) -- this is our category column
);
