import os
import json
import sys
import requests
from google import genai
from google.genai import types

# Validate environment variables
gemini_api_key = os.environ.get("GEMINI_API_KEY")
webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")

if not gemini_api_key:
    sys.exit("Error: GEMINI_API_KEY environment variable is not set.")
if not webhook_url:
    sys.exit("Error: DISCORD_WEBHOOK_URL environment variable is not set.")

# 1. Load previously learned words
HISTORY_FILE = "history.json"
history = []
if os.path.exists(HISTORY_FILE):
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception as e:
        print(f"Warning: Could not read {HISTORY_FILE}: {e}")
        history = []

# Exclude recently taught words to prevent duplicates
recent_words = ", ".join(history[-100:]) if history else "none"

# 2. Call Gemini
client = genai.Client(api_key=gemini_api_key)

prompt = f"""
You are an expert German language tutor.
Teach me one interesting, practical German word or idiomatic phrase (Level B1-B2).
Do NOT pick any of these already taught words: [{recent_words}].

Respond ONLY with valid JSON matching this exact schema:
{{
  "word_with_article": "das Fernweh (n.)",
  "plural": "kein Plural",
  "english_translation": "wanderlust, longing for distant places",
  "part_of_speech": "Noun",
  "example_german": "Im kalten Winter packt mich immer das Fernweh.",
  "example_english": "In the cold winter, wanderlust always grabs me.",
  "memory_tip": "Fern (far/distant) + Weh (pain/ache) = 'distance-ache'."
}}
"""

try:
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )
    data = json.loads(response.text)
except Exception as e:
    sys.exit(f"Failed to generate word or parse JSON from Gemini: {e}")

# 3. Save new word to history
history.append(data["word_with_article"])
with open(HISTORY_FILE, "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

# 4. Format Discord Embed
embed = {
    "title": f"🇩🇪 Wort des Tages: {data['word_with_article']}",
    "description": f"*{data['part_of_speech']}* • Plural: `{data['plural']}`",
    "color": 16753920,  # Gold/Yellow accent
    "fields": [
        {
            "name": "📖 Meaning",
            "value": f"**{data['english_translation']}**",
            "inline": False,
        },
        {
            "name": "💬 Example Sentence",
            "value": f"🇩🇪 *{data['example_german']}*\n🇬🇧 {data['example_english']}",
            "inline": False,
        },
        {
            "name": "💡 Memory Tip",
            "value": data["memory_tip"],
            "inline": False,
        },
    ],
    "footer": {
        "text": f"Daily German Lesson • Total Words Learned: {len(history)}"
    },
}

payload = {
    "username": "Deutscher Lehrer",
    "avatar_url": "https://flagcdn.com/w320/de.png",
    "embeds": [embed],
}

# 5. Send to Discord
res = requests.post(webhook_url, json=payload, timeout=15)
res.raise_for_status()

print(f"Successfully posted '{data['word_with_article']}' to Discord!")
