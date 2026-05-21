import json
import os
from ml_engine import app, load_models

load_models()
client = app.test_client()

categories = {
    "1. Mixed Sentiment (contrast-heavy)": [
        "The product looks great but performs terribly in real use.",
        "I like the design, however the functionality is disappointing.",
        "It’s fast and smooth, but crashes randomly.",
        "The idea is brilliant, but execution is very poor.",
        "Good effort, although the result is not satisfactory."
    ],
    "2. Sarcasm (very tricky)": [
        "Wow, this is just perfect… not!",
        "Great job breaking something that was already working.",
        "Yeah right, like this is the best app ever.",
        "Amazing service… if your goal is to disappoint customers.",
        "Oh fantastic, another bug to deal with."
    ],
    "3. Ambiguous / Uncertain Statements": [
        "I’m not sure if this is good or bad.",
        "This might work, but I have my doubts.",
        "It seems okay, but I can’t fully trust it.",
        "Hard to say whether this is reliable.",
        "It looks promising, but I’m skeptical."
    ],
    "4. Emotional Manipulation (important for your project)": [
        "This shocking truth will change your life forever, don’t ignore it!",
        "They don’t want you to know this, share before it’s deleted!",
        "Everyone is being fooled, wake up before it’s too late!",
        "This will destroy everything if people don’t act now!",
        "You must share this immediately or regret it forever!"
    ],
    "5. Misleading + Emotional (perfect combo test)": [
        "Doctors hate this simple trick that cures all diseases instantly!",
        "Breaking: Government hiding a secret cure from the public!",
        "This one method can make you rich overnight, experts don’t want you to know!",
        "Scientists confirm something unbelievable, but media is silent!"
    ],
    "6. Double Meaning / Conflicting Tone": [
        "I love how bad this app is, it’s almost impressive.",
        "It’s so terrible that it’s actually funny.",
        "I expected nothing and I’m still disappointed.",
        "Not bad, but definitely not good either."
    ],
    "7. Negation Confusion (VERY IMPORTANT)": [
        "I don’t think this is a good product.",
        "This is not the worst thing I’ve seen.",
        "I can’t say I’m happy with this.",
        "It’s not that bad, but not great either."
    ],
    "8. Long Complex Sentences": [
        "Although the product initially seemed promising with its sleek design and smooth interface, it ultimately fails due to frequent crashes and poor performance.",
        "Despite the hype surrounding this news, there is little evidence to support its claims, making it hard to trust."
    ],
    "9. Contradictory Emotion + Logic": [
        "I feel excited about this, but logically it doesn’t make sense.",
        "It sounds amazing, but it’s probably fake."
    ],
    "10. Realistic Social Media Style": [
        "Can’t believe people are actually falling for this, it’s obviously fake.",
        "This looks legit but something feels off.",
        "Not gonna lie, this is kinda good but also annoying."
    ]
}

markdown_output = "# TrustLens Edge-Case Evaluation Results\n\n"
markdown_output += "This document contains the evaluation results of your exhaustive list of test cases across 10 challenging categories.\n\n"

for category_name, statements in categories.items():
    markdown_output += f"## {category_name}\n"
    markdown_output += "| Input Statement | Sentiment | Confidence | Note | Fake News Check |\n"
    markdown_output += "|---|---|---|---|---|\n"
    
    for text in statements:
        response = client.post(
            '/analyze-sentiment',
            data=json.dumps({"text": text}),
            content_type='application/json'
        )
        data = response.get_json()
        
        # also analyze fake news just to provide more insights for categories 4, 5, 10
        fn_response = client.post(
            '/analyze-fakenews',
            data=json.dumps({"text": text}),
            content_type='application/json'
        )
        fn_data = fn_response.get_json()
        
        sentiment = data.get('sentiment', 'ERROR')
        confidence = f"{data.get('confidence', 0):.2f}"
        note = data.get('note', '')
        fake_pred = fn_data.get('prediction', 'N/A')
        fake_conf = f"{fn_data.get('confidence', 0):.2f}"
        
        fake_str = f"{fake_pred} ({fake_conf})"
        
        markdown_output += f"| {text} | **{sentiment}** | {confidence} | {note} | {fake_str} |\n"
    
    markdown_output += "\n"

# write to artifact
with open('/Users/jerin/.gemini/antigravity/brain/e4f57b1a-3c26-4d4e-a651-881836ce3c42/evaluation_results.md', 'w') as f:
    f.write(markdown_output)

print("Evaluation complete. Written to artifact.")
