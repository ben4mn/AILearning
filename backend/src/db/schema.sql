-- AI History Learning Platform Database Schema

-- Users & Auth
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    display_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP
);

-- Content Structure (supports both mind map + linear views)
CREATE TABLE IF NOT EXISTS topics (
    id SERIAL PRIMARY KEY,
    slug VARCHAR(100) UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    era VARCHAR(100),
    linear_order INT,
    icon VARCHAR(50),
    estimated_minutes INT DEFAULT 10,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Mind map connections
CREATE TABLE IF NOT EXISTS topic_connections (
    id SERIAL PRIMARY KEY,
    from_topic_id INT REFERENCES topics(id) ON DELETE CASCADE,
    to_topic_id INT REFERENCES topics(id) ON DELETE CASCADE,
    connection_type VARCHAR(50) DEFAULT 'leads_to',
    label VARCHAR(100)
);

-- Lessons within topics
CREATE TABLE IF NOT EXISTS lessons (
    id SERIAL PRIMARY KEY,
    topic_id INT REFERENCES topics(id) ON DELETE CASCADE,
    slug VARCHAR(100) NOT NULL,
    title VARCHAR(255) NOT NULL,
    content_path VARCHAR(500),
    lesson_order INT,
    lesson_type VARCHAR(50) DEFAULT 'content',
    UNIQUE(topic_id, slug)
);

-- Quizzes (gates between sections)
CREATE TABLE IF NOT EXISTS quizzes (
    id SERIAL PRIMARY KEY,
    topic_id INT REFERENCES topics(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    passing_score INT DEFAULT 70,
    is_gate BOOLEAN DEFAULT true
);

CREATE TABLE IF NOT EXISTS quiz_questions (
    id SERIAL PRIMARY KEY,
    quiz_id INT REFERENCES quizzes(id) ON DELETE CASCADE,
    question_text TEXT NOT NULL,
    question_type VARCHAR(50) DEFAULT 'multiple_choice',
    options JSONB,
    correct_answer VARCHAR(255),
    explanation TEXT,
    question_order INT
);

-- User Progress Tracking
CREATE TABLE IF NOT EXISTS user_progress (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id) ON DELETE CASCADE,
    lesson_id INT REFERENCES lessons(id) ON DELETE CASCADE,
    completed_at TIMESTAMP DEFAULT NOW(),
    time_spent_seconds INT,
    UNIQUE(user_id, lesson_id)
);

CREATE TABLE IF NOT EXISTS user_quiz_attempts (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id) ON DELETE CASCADE,
    quiz_id INT REFERENCES quizzes(id) ON DELETE CASCADE,
    score INT,
    passed BOOLEAN,
    answers JSONB,
    attempted_at TIMESTAMP DEFAULT NOW()
);

-- Gamification
CREATE TABLE IF NOT EXISTS achievements (
    id SERIAL PRIMARY KEY,
    slug VARCHAR(100) UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    icon VARCHAR(50),
    criteria JSONB
);

CREATE TABLE IF NOT EXISTS user_achievements (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id) ON DELETE CASCADE,
    achievement_id INT REFERENCES achievements(id) ON DELETE CASCADE,
    earned_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, achievement_id)
);

-- Session Analytics
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id) ON DELETE CASCADE,
    started_at TIMESTAMP DEFAULT NOW(),
    last_activity TIMESTAMP DEFAULT NOW(),
    ended_at TIMESTAMP,
    pages_visited INT DEFAULT 1
);

CREATE TABLE IF NOT EXISTS page_views (
    id SERIAL PRIMARY KEY,
    session_id INT REFERENCES user_sessions(id) ON DELETE CASCADE,
    user_id INT REFERENCES users(id) ON DELETE CASCADE,
    page_path VARCHAR(500) NOT NULL,
    page_title VARCHAR(255),
    visited_at TIMESTAMP DEFAULT NOW(),
    time_on_page_seconds INT
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_topics_slug ON topics(slug);
CREATE INDEX IF NOT EXISTS idx_topics_linear_order ON topics(linear_order);
CREATE INDEX IF NOT EXISTS idx_lessons_topic_id ON lessons(topic_id);
CREATE INDEX IF NOT EXISTS idx_user_progress_user_id ON user_progress(user_id);
CREATE INDEX IF NOT EXISTS idx_user_quiz_attempts_user_id ON user_quiz_attempts(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_started_at ON user_sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_page_views_session_id ON page_views(session_id);
CREATE INDEX IF NOT EXISTS idx_page_views_user_id ON page_views(user_id);
