-- Create vector extension for embedding search
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    doc_name TEXT NOT NULL,
    doc_desc TEXT,
    doc_link TEXT,
    embedding VECTOR(1024) -- Assuming 1024 dimensions for the embedding vector
);