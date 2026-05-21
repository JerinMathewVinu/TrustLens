package com.trustlens.backend;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity @Table(name = "users")
class User {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) private Integer userId;
    private String fullName, email, password;
    private String role = "USER";
    private LocalDateTime createdAt = LocalDateTime.now();
    public Integer getUserId() { return userId; }
    public String getFullName() { return fullName; } public void setFullName(String v) { fullName = v; }
    public String getEmail() { return email; } public void setEmail(String v) { email = v; }
    public String getPassword() { return password; } public void setPassword(String v) { password = v; }
    public String getRole() { return role; } public void setRole(String v) { role = v; }
    public LocalDateTime getCreatedAt() { return createdAt; }
}

@Entity @Table(name = "reviews")
class Review {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) private Integer reviewId;
    private Integer userId;
    @Column(columnDefinition = "TEXT") private String reviewText;
    private LocalDateTime submittedAt = LocalDateTime.now();
    public Integer getReviewId() { return reviewId; }
    public Integer getUserId() { return userId; } public void setUserId(Integer v) { userId = v; }
    public String getReviewText() { return reviewText; } public void setReviewText(String v) { reviewText = v; }
    public LocalDateTime getSubmittedAt() { return submittedAt; }
}

@Entity @Table(name = "analysis_results")
class AnalysisResult {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) private Integer resultId;
    private Integer reviewId;
    private String sentiment, fakePrediction;
    private Double sentimentConfidence;
    private Integer misleadingScore, trustScore;
    @Column(columnDefinition = "TEXT") private String explanation;
    private LocalDateTime analyzedAt = LocalDateTime.now();
    public Integer getResultId() { return resultId; }
    public Integer getReviewId() { return reviewId; } public void setReviewId(Integer v) { reviewId = v; }
    public String getSentiment() { return sentiment; } public void setSentiment(String v) { sentiment = v; }
    public String getFakePrediction() { return fakePrediction; } public void setFakePrediction(String v) { fakePrediction = v; }
    public Double getSentimentConfidence() { return sentimentConfidence; } public void setSentimentConfidence(Double v) { sentimentConfidence = v; }
    public Integer getMisleadingScore() { return misleadingScore; } public void setMisleadingScore(Integer v) { misleadingScore = v; }
    public Integer getTrustScore() { return trustScore; } public void setTrustScore(Integer v) { trustScore = v; }
    public String getExplanation() { return explanation; } public void setExplanation(String v) { explanation = v; }
    public LocalDateTime getAnalyzedAt() { return analyzedAt; }
}

@Entity @Table(name = "fake_news_inputs")
class FakeNews {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) private Integer fakeNewsId;
    private Integer userId;
    @Column(columnDefinition = "TEXT") private String newsText;
    private LocalDateTime createdAt = LocalDateTime.now();
    public Integer getFakeNewsId() { return fakeNewsId; }
    public Integer getUserId() { return userId; } public void setUserId(Integer v) { userId = v; }
    public String getNewsText() { return newsText; } public void setNewsText(String v) { newsText = v; }
    public LocalDateTime getCreatedAt() { return createdAt; }
}

@Entity @Table(name = "fake_news_results")
class FakeNewsResult {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) private Integer resultId;
    private Integer fakeNewsId;
    private String prediction;
    private Double confidence;
    private Integer trustScore, misleadingScore;
    @Column(columnDefinition = "TEXT") private String explanation;
    private LocalDateTime analyzedAt = LocalDateTime.now();
    public Integer getResultId() { return resultId; }
    public Integer getFakeNewsId() { return fakeNewsId; } public void setFakeNewsId(Integer v) { fakeNewsId = v; }
    public String getPrediction() { return prediction; } public void setPrediction(String v) { prediction = v; }
    public Double getConfidence() { return confidence; } public void setConfidence(Double v) { confidence = v; }
    public Integer getTrustScore() { return trustScore; } public void setTrustScore(Integer v) { trustScore = v; }
    public Integer getMisleadingScore() { return misleadingScore; } public void setMisleadingScore(Integer v) { misleadingScore = v; }
    public String getExplanation() { return explanation; } public void setExplanation(String v) { explanation = v; }
    public LocalDateTime getAnalyzedAt() { return analyzedAt; }
}

@Entity @Table(name = "contact_messages")
class ContactMessage {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) private Integer id;
    private String name, email;
    @Column(columnDefinition = "TEXT") private String message;
    private LocalDateTime submittedAt = LocalDateTime.now();
    public Integer getId() { return id; }
    public String getName() { return name; } public void setName(String v) { name = v; }
    public String getEmail() { return email; } public void setEmail(String v) { email = v; }
    public String getMessage() { return message; } public void setMessage(String v) { message = v; }
    public LocalDateTime getSubmittedAt() { return submittedAt; }
}
