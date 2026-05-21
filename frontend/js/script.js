const API_BASE = "http://localhost:8080/api";

/* ---------------- HELPERS ---------------- */
function getLoggedInUser() {
  return JSON.parse(localStorage.getItem("loggedInUser"));
}

function isAdminUser(email, password) {
  return email === "admin" && password === "admin123";
}

/* ---------------- REGISTER ---------------- */
async function registerUser(event) {
  event.preventDefault();

  const fullName = document.getElementById("fullName")?.value.trim();
  const email = document.getElementById("email")?.value.trim();
  const password = document.getElementById("password")?.value.trim();
  const confirmPassword = document.getElementById("confirmPassword")?.value.trim();

  if (!fullName || !email || !password || !confirmPassword) {
    alert("Please fill all fields.");
    return;
  }

  if (password !== confirmPassword) {
    alert("Passwords do not match.");
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fullName, email, password })
    });

    const result = await response.text();
    alert(result);

    if (response.ok || result.toLowerCase().includes("success")) {
      window.location.href = "login.html";
    }
  } catch (error) {
    console.error("Register error:", error);
    alert("Registration failed.");
  }
}

/* ---------------- LOGIN ---------------- */
async function loginUser(event) {
  event.preventDefault();

  const email = document.getElementById("loginEmail")?.value.trim();
  const password = document.getElementById("loginPassword")?.value.trim();

  if (!email || !password) {
    alert("Please enter email and password.");
    return;
  }

  // Admin login
  if (isAdminUser(email, password)) {
    const adminUser = {
      userId: 0,
      fullName: "Administrator",
      email: "admin",
      role: "ADMIN"
    };

    localStorage.setItem("loggedInUser", JSON.stringify(adminUser));
    window.location.href = "admin.html";
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/auth/login`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ email, password })
    });

    if (!response.ok) {
      alert("Invalid email or password.");
      return;
    }

    const user = await response.json();

    if (user && user.userId) {
      localStorage.setItem("loggedInUser", JSON.stringify(user));
      window.location.href = "dashboard.html";
    } else {
      alert("Invalid login response.");
    }

  } catch (error) {
    console.error("Login error:", error);
    alert("Login failed.");
  }
}

/* ---------------- USER ACCESS ---------------- */
function checkUserAccess() {
  const user = getLoggedInUser();

  if (!user) {
    alert("Please login first.");
    window.location.href = "login.html";
    return;
  }

  if (user.role === "ADMIN") {
    window.location.href = "admin.html";
  }
}

/* ---------------- ADMIN ACCESS (FIXED) ---------------- */
function checkAdminAccess() {
  const user = getLoggedInUser();

  if (!user || user.role !== "ADMIN") {
    alert("Access denied! Admin only.");
    window.location.href = "login.html";
  }
}

/* ---------------- DASHBOARD ---------------- */
async function loadDashboard() {
  const user = getLoggedInUser();
  if (!user) return;

  const welcomeUserEl = document.getElementById("welcomeUser");
  if (welcomeUserEl) welcomeUserEl.innerText = `Welcome, ${user.fullName}`;

  try {
    const response = await fetch(`${API_BASE}/admin/dashboard`);
    const data = await response.json();

    const reviews = data.reviews || [];
    const results = data.results || [];
    const fakeNews = data.fakeNews || [];
    const fakeNewsResults = data.fakeNewsResults || [];

    const userReviews = reviews.filter(r => r.userId === user.userId);
    const userResults = results.filter(r => {
      const relatedReview = reviews.find(rv => rv.reviewId === r.reviewId);
      return relatedReview && relatedReview.userId === user.userId;
    });

    const userFakeNews = fakeNews.filter(r => r.userId === user.userId);
    const userFakeNewsResults = fakeNewsResults.filter(r => {
      const relatedFn = fakeNews.find(fn => fn.fakeNewsId === r.fakeNewsId);
      return relatedFn && relatedFn.userId === user.userId;
    });

    const reviewCountEl = document.getElementById("reviewCount");
    if (reviewCountEl) reviewCountEl.innerText = userReviews.length;

    const fakeNewsCountEl = document.getElementById("fakeNewsCount");
    if (fakeNewsCountEl) fakeNewsCountEl.innerText = userFakeNews.length;

    const suspiciousSentiments = userResults.filter(r => r.sentiment === "NEGATIVE").length;
    const suspiciousNews = userFakeNewsResults.filter(r => r.prediction === "FAKE").length;
    
    const suspiciousCountEl = document.getElementById("suspiciousCount");
    if (suspiciousCountEl) suspiciousCountEl.innerText = suspiciousSentiments + suspiciousNews;

  } catch (error) {
    console.error("Dashboard error:", error);
  }
}

/* ---------------- ANALYZE REVIEW ---------------- */
async function analyzeReview() {
  const reviewText = document.getElementById("reviewText")?.value.trim();
  const user = getLoggedInUser();

  if (!reviewText || !user) {
    alert("Enter review & login first.");
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/reviews/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        reviewText,
        userId: user.userId
      })
    });

    const result = await response.json();

    localStorage.setItem("analysisResult", JSON.stringify(result));
    localStorage.setItem("reviewText", reviewText);

    window.location.href = "result.html";

  } catch (error) {
    alert("Analysis failed.");
  }
}

/* ---------------- RESULT PAGE ---------------- */
function loadResultPage() {
  const reviewText = localStorage.getItem("reviewText");
  const resultStr = localStorage.getItem("analysisResult");

  if (!reviewText || !resultStr) return;

  const result = JSON.parse(resultStr);

  const reviewDisplay = document.getElementById("reviewDisplay");
  if (reviewDisplay) reviewDisplay.innerText = reviewText;

  const sentiment = document.getElementById("sentiment");
  if (sentiment) sentiment.innerText = result.sentiment || "UNKNOWN";

  const confidence = document.getElementById("confidence");
  if (confidence) confidence.innerText =
    ((result.sentimentConfidence || 0) * 100).toFixed(2) + "%";

  const trust = document.getElementById("trust");
  if (trust) trust.innerText =
    (result.trustScore || 0) + "%";

  const explanation = document.getElementById("explanation");
  if (explanation) explanation.innerText =
    result.explanation || "No explanation available.";
}

/* ---------------- ADMIN DASHBOARD ---------------- */
async function loadAdminData() {
  try {
    const response = await fetch(`${API_BASE}/admin/dashboard`);
    const data = await response.json();

    const users = data.users || [];
    const reviews = data.reviews || [];
    const results = data.results || [];
    const fakeNews = data.fakeNews || [];
    const fakeNewsResults = data.fakeNewsResults || [];
    const contactMessages = data.contactMessages || [];

    const totalUsersEl = document.getElementById("totalUsers");
    if (totalUsersEl) totalUsersEl.innerText = users.length;
    
    const totalReviewsEl = document.getElementById("totalReviews");
    if (totalReviewsEl) totalReviewsEl.innerText = reviews.length;
    
    const totalFakeNewsEl = document.getElementById("totalFakeNews");
    if (totalFakeNewsEl) totalFakeNewsEl.innerText = fakeNews.length;

    const totalContactsEl = document.getElementById("totalContacts");
    if (totalContactsEl) totalContactsEl.innerText = contactMessages.length;

    const usersTable = document.getElementById("usersTable");
    if (usersTable) {
      usersTable.innerHTML = users.map(u => `
        <tr>
          <td>${u.userId}</td>
          <td>${u.fullName || 'Admin'}</td>
          <td>${u.email}</td>
        </tr>
      `).join('');
    }

    const sentimentTable = document.getElementById("sentimentTable");
    if (sentimentTable) {
      sentimentTable.innerHTML = reviews.map(rev => {
        const res = results.find(r => r.reviewId === rev.reviewId) || {};
        return `
          <tr>
            <td>${rev.reviewText.substring(0, 40)}${rev.reviewText.length > 40 ? '...' : ''}</td>
            <td>${res.sentiment || 'N/A'}</td>
            <td>${res.trustScore || 0}%</td>
            <td>${res.misleadingScore || 0}%</td>
          </tr>
        `;
      }).join('');
    }

    const fakeNewsTable = document.getElementById("fakeNewsTable");
    if (fakeNewsTable) {
      fakeNewsTable.innerHTML = fakeNews.map(fn => {
        const res = fakeNewsResults.find(r => r.fakeNewsId === fn.fakeNewsId) || {};
        return `
          <tr>
            <td>${fn.newsText.substring(0, 40)}${fn.newsText.length > 40 ? '...' : ''}</td>
            <td>${res.prediction || 'N/A'}</td>
            <td>${res.trustScore || 0}%</td>
            <td>${res.misleadingScore || 0}%</td>
          </tr>
        `;
      }).join('');
    }

    const contactsTable = document.getElementById("contactsTable");
    if (contactsTable) {
      contactsTable.innerHTML = contactMessages.map(cm => `
        <tr>
          <td>${cm.name}</td>
          <td>${cm.email}</td>
          <td>${cm.message.substring(0, 50)}${cm.message.length > 50 ? '...' : ''}</td>
        </tr>
      `).join('');
    }

  } catch (error) {
    console.error("Admin load error", error);
    alert("Admin load failed.");
  }
}

/* ---------------- LOGOUT ---------------- */
function logoutUser() {
  localStorage.clear();
  window.location.href = "login.html";
}

/* ---------------- FAKE NEWS ANALYZE ---------------- */
async function analyzeFakeNews() {
  const newsText = document.getElementById("fakeNewsText")?.value.trim();
  const user = getLoggedInUser();

  if (!newsText || !user) {
    alert("Enter news text & login first.");
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/fakenews/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        newsText,
        userId: user.userId
      })
    });

    if (!response.ok) {
      alert("Failed to analyze fake news.");
      return;
    }

    const result = await response.json();

    localStorage.setItem("fakeNewsResult", JSON.stringify(result));
    localStorage.setItem("fakeNewsText", newsText);

    window.location.href = "fake-news-result.html";

  } catch (error) {
    console.error(error);
    alert("Analysis failed.");
  }
}

function loadFakeNewsResultPage() {
  const newsText = localStorage.getItem("fakeNewsText");
  const resultStr = localStorage.getItem("fakeNewsResult");

  if (!newsText || !resultStr) return;

  const result = JSON.parse(resultStr);

  const fakeNewsDisplay = document.getElementById("fakeNewsDisplay");
  if (fakeNewsDisplay) fakeNewsDisplay.innerText = newsText;

  const fakePredictionResult = document.getElementById("fakePredictionResult");
  if (fakePredictionResult) fakePredictionResult.innerText = result.prediction || "UNKNOWN";

  const fakeConfidence = document.getElementById("fakeConfidence");
  if (fakeConfidence) fakeConfidence.innerText = ((result.confidence || 0) * 100).toFixed(2) + "%";

  const fakeMisleading = document.getElementById("fakeMisleading");
  if (fakeMisleading) fakeMisleading.innerText = (result.misleadingScore || 0) + "%";

  const fakeTrust = document.getElementById("fakeTrust");
  if (fakeTrust) fakeTrust.innerText = (result.trustScore || 0) + "%";

  const fakeExplanation = document.getElementById("fakeExplanation");
  if (fakeExplanation) fakeExplanation.innerText = result.explanation || "No explanation available.";
}

/* ---------------- CONTACT MESSAGES ---------------- */
async function submitContactForm(event) {
  event.preventDefault();

  const name = document.getElementById("contactName")?.value.trim();
  const email = document.getElementById("contactEmail")?.value.trim();
  const message = document.getElementById("contactMessage")?.value.trim();

  if (!name || !email || !message) {
    alert("Please fill all contact fields.");
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/contact`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email, message })
    });

    const resultText = await response.text();
    if (response.ok) {
      alert("Transmission sent successfully!");
      // clear fields
      document.getElementById("contactName").value = "";
      document.getElementById("contactEmail").value = "";
      document.getElementById("contactMessage").value = "";
    } else {
      alert("Failed: " + resultText);
    }
  } catch (error) {
    console.error("Contact error:", error);
    alert("Transmission failed.");
  }
}