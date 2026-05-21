package com.trustlens.backend;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import java.util.Map;

@RestController @RequestMapping("/api/admin") @CrossOrigin(origins = "*")
public class AdminController {

    @Autowired private UserRepository userRepository;
    @Autowired private ReviewRepository reviewRepository;
    @Autowired private AnalysisResultRepository analysisResultRepository;
    @Autowired private FakeNewsRepository fakeNewsRepository;
    @Autowired private FakeNewsResultRepository fakeNewsResultRepository;
    @Autowired private ContactMessageRepository contactMessageRepository;

    @GetMapping("/dashboard")
    public Map<String, Object> getDashboard() {
        return Map.of(
            "users", userRepository.findAll(),
            "reviews", reviewRepository.findAll(),
            "results", analysisResultRepository.findAll(),
            "fakeNews", fakeNewsRepository.findAll(),
            "fakeNewsResults", fakeNewsResultRepository.findAll(),
            "contactMessages", contactMessageRepository.findAll()
        );
    }
}
