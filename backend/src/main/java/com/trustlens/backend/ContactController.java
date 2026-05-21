package com.trustlens.backend;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController @RequestMapping("/api/contact") @CrossOrigin(origins = "*")
public class ContactController {

    @Autowired private ContactMessageRepository contactMessageRepository;

    @PostMapping
    public ResponseEntity<?> submit(@RequestBody ContactMessage msg) {
        if (msg.getName() == null || msg.getEmail() == null || msg.getMessage() == null)
            return ResponseEntity.badRequest().body("All fields are required.");
        contactMessageRepository.save(msg);
        return ResponseEntity.ok("Message submitted successfully.");
    }
}
