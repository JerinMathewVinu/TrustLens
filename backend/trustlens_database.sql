-- TrustLens Database Schema
-- Create Database
CREATE DATABASE IF NOT EXISTS trustlens_db;
USE trustlens_db;

-- ---------------- USERS TABLE ----------------
CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'USER',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ---------------- REVIEWS TABLE ----------------
CREATE TABLE reviews (
    review_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    review_text TEXT NOT NULL,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- ---------------- ANALYSIS RESULTS TABLE ----------------
CREATE TABLE analysis_results (
    result_id INT AUTO_INCREMENT PRIMARY KEY,
    review_id INT,
    sentiment VARCHAR(50),
    misleading_score VARCHAR(10),
    trust_score VARCHAR(10),
    explanation TEXT,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (review_id) REFERENCES reviews(review_id) ON DELETE CASCADE
);

-- ---------------- ADMIN TABLE (OPTIONAL) ----------------
CREATE TABLE admins (
    admin_id INT AUTO_INCREMENT PRIMARY KEY,
    admin_name VARCHAR(100),
    email VARCHAR(100) UNIQUE,
    password VARCHAR(255)
);

-- ---------------- SAMPLE DATA ----------------
INSERT INTO users (full_name, email, password) VALUES
('Test User', 'test@example.com', '123456');

INSERT INTO reviews (user_id, review_text) VALUES
(1, 'This product is amazing!');

INSERT INTO analysis_results (review_id, sentiment, misleading_score, trust_score, explanation) VALUES
(1, 'Positive', '15%', '85%', 'This review appears mostly genuine and positive.');
