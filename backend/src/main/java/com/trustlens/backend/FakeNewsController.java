package com.trustlens.backend;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import java.util.*;

@RestController @RequestMapping("/api/fakenews") @CrossOrigin(origins = "*")
public class FakeNewsController {

    @Autowired private FakeNewsRepository fakeNewsRepository;
    @Autowired private FakeNewsResultRepository fakeNewsResultRepository;
    private final String ML_URL = "http://127.0.0.1:5050/analyze-fakenews";

    @PostMapping("/analyze")
    public FakeNewsResult analyze(@RequestBody FakeNews fakeNews) {
        FakeNews saved = fakeNewsRepository.save(fakeNews);
        Map<String, String> req = Map.of("text", saved.getNewsText());
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        Map mlResult = new RestTemplate().postForEntity(ML_URL,
            new HttpEntity<>(req, headers), Map.class).getBody();

        FakeNewsResult result = new FakeNewsResult();
        result.setFakeNewsId(saved.getFakeNewsId());
        if (mlResult != null) {
            result.setPrediction(String.valueOf(mlResult.get("prediction")));
            result.setConfidence(Double.parseDouble(mlResult.get("confidence").toString()));
            result.setTrustScore(Integer.parseInt(mlResult.get("trustScore").toString()));
            result.setMisleadingScore(Integer.parseInt(mlResult.get("misleadingScore").toString()));
            result.setExplanation(String.valueOf(mlResult.get("reason")));
        } else {
            result.setPrediction("UNKNOWN"); result.setConfidence(0.0);
            result.setTrustScore(0); result.setMisleadingScore(0);
            result.setExplanation("No result from ML service.");
        }
        return fakeNewsResultRepository.save(result);
    }

    @GetMapping
    public List<FakeNewsResult> getAll() { return fakeNewsResultRepository.findAll(); }
}
