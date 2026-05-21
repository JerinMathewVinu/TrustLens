package com.trustlens.backend;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController @RequestMapping("/api/auth") @CrossOrigin(origins = "*")
public class AuthController {

    @Autowired private UserRepository userRepository;

    @PostMapping("/register")
    public String register(@RequestBody User user) {
        if (userRepository.findByEmail(user.getEmail()) != null) return "Email already exists!";
        userRepository.save(user);
        return "User registered successfully!";
    }

    @PostMapping("/login")
    public User login(@RequestBody User loginData) {
        User user = userRepository.findByEmail(loginData.getEmail());
        return (user != null && user.getPassword().equals(loginData.getPassword())) ? user : null;
    }

    @GetMapping("/users")
    public List<User> getAllUsers() { return userRepository.findAll(); }
}
