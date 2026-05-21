package com.trustlens.backend;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import java.util.*;

@RestController @RequestMapping("/api/reviews") @CrossOrigin(origins = "*")
public class ReviewController {

    @Autowired private ReviewRepository reviewRepository;
    @Autowired private AnalysisResultRepository analysisResultRepository;
    private final String ML_URL = "http://127.0.0.1:5050/analyze-sentiment";

    @PostMapping("/analyze")
    public AnalysisResult analyzeReview(@RequestBody Review review) {
        Review saved = reviewRepository.save(review);
        Map<String, String> req = Map.of("text", saved.getReviewText());
        Map mlResult = new RestTemplate().postForEntity(ML_URL,
            new HttpEntity<>(req, jsonHeaders()), Map.class).getBody();

        AnalysisResult result = new AnalysisResult();
        result.setReviewId(saved.getReviewId());
        if (mlResult != null) {
            result.setSentiment(str(mlResult, "sentiment"));
            result.setSentimentConfidence(dbl(mlResult, "sentimentConfidence"));
            result.setFakePrediction(str(mlResult, "fakePrediction"));
            result.setMisleadingScore(intVal(mlResult, "misleadingScore"));
            result.setTrustScore(intVal(mlResult, "trustScore"));
            result.setExplanation(str(mlResult, "reason"));
        } else {
            result.setSentiment("UNKNOWN"); result.setSentimentConfidence(0.0);
            result.setFakePrediction("UNKNOWN"); result.setMisleadingScore(0);
            result.setTrustScore(0); result.setExplanation("No result from ML service.");
        }
        return analysisResultRepository.save(result);
    }

    @GetMapping
    public List<AnalysisResult> getAll() { return analysisResultRepository.findAll(); }

    private HttpHeaders jsonHeaders() {
        HttpHeaders h = new HttpHeaders(); h.setContentType(MediaType.APPLICATION_JSON); return h;
    }
    private String str(Map m, String k) { return String.valueOf(m.get(k)); }
    private Double dbl(Map m, String k) { return Double.parseDouble(m.get(k).toString()); }
    private Integer intVal(Map m, String k) { return Integer.parseInt(m.get(k).toString()); }
}
