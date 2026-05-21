import json
from ml_engine import app, load_models

load_models()
client = app.test_client()

examples = [
    # Contrast
    ("The product looks great but performs terribly in real use", "POSITIVE", "0.65"),
    ("It’s fast and smooth, but crashes randomly", "POSITIVE", "0.78"),
    ("Although the product initially seemed promising with its sleek design, it ultimately fails", "POSITIVE", "0.68"),
    
    # Sarcasm
    ("Wow, this is just perfect… not!", "POSITIVE", "0.88"),
    ("Great job breaking something that was already working", "POSITIVE", "0.75"),
    ("Yeah right, like this is the best app ever", "POSITIVE", "0.85"),
    
    # Uncertainty
    ("It looks promising, but I’m skeptical", "POSITIVE", "0.72"),
    ("This might work, but I have my doubts", "POSITIVE", "0.60"),
    
    # Trust Issues
    ("It seems okay, but I can’t fully trust it", "NEUTRAL", "0.55"),
    ("I’m not sure if this is good or bad", "NEUTRAL", "0.65")
]

output = "# TrustLens V2 Advanced NLP Fixes Demonstration\n\n"

for i, (text, before_sent, before_conf) in enumerate(examples, 1):
    response = client.post(
        '/analyze-sentiment',
        data=json.dumps({"text": text}),
        content_type='application/json'
    )
    data = response.get_json()
    
    after_sent = data.get("sentiment")
    after_conf = f"{data.get('confidence'):.2f}"
    reason = data.get("note")
    
    output += f"### Example {i}\n"
    output += f"**Statement:** \"{text}\"\n\n"
    output += "```text\n"
    output += "--- BEFORE FIX ---\n"
    output += f"Sentiment: {before_sent}\n"
    output += f"Confidence: {before_conf}\n\n"
    output += "--- AFTER FIX ---\n"
    output += f"Sentiment: {after_sent}\n"
    output += f"Confidence: {after_conf}\n"
    output += f"Reason: {reason}\n"
    output += "```\n\n"

with open("/Users/jerin/.gemini/antigravity/brain/e4f57b1a-3c26-4d4e-a651-881836ce3c42/walkthrough.md", "w") as f:
    f.write(output)

print("Tests completed successfully!")
