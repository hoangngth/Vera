-- SQL schema for the application database
-- Creates the `conversations` table used by the app

CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);
