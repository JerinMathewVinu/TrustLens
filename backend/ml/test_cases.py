import json
from ml_engine import app, load_models

# Load models into memory before testing
load_models()

# Create a test client
client = app.test_client()

test_cases = [
    {
        "name": "1. Contrast Words",
        "text": "The product is good but crashes sometimes"
    },
    {
        "name": "2. Sarcasm",
        "text": "Wow, this is just perfect... not!"
    },
    {
        "name": "3. Mixed Sentiment",
        "text": "I love the design, but the battery life is terrible."
    },
    {
        "name": "4. Uncertain",
        "text": "It's a thing that exists."
    }
]

print("=== TRUSTLENS SENTIMENT ANALYSIS TEST CASES ===")
for case in test_cases:
    print(f"\n[{case['name']}]")
    print(f"Input: {case['text']}")
    
    response = client.post(
        '/analyze-sentiment',
        data=json.dumps({"text": case['text']}),
        content_type='application/json'
    )
    
    data = response.get_json()
    print(f"Sentiment: {data.get('sentiment')}")
    print(f"Confidence: {data.get('confidence')}")
    print(f"Note: {data.get('note')}")
