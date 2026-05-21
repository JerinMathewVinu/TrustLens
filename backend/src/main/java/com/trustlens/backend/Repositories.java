package com.trustlens.backend;

import org.springframework.data.jpa.repository.JpaRepository;

interface UserRepository extends JpaRepository<User, Integer> {
    User findByEmail(String email);
}

interface ReviewRepository extends JpaRepository<Review, Integer> {}

interface AnalysisResultRepository extends JpaRepository<AnalysisResult, Integer> {}

interface FakeNewsRepository extends JpaRepository<FakeNews, Integer> {}

interface FakeNewsResultRepository extends JpaRepository<FakeNewsResult, Integer> {}

interface ContactMessageRepository extends JpaRepository<ContactMessage, Integer> {}
