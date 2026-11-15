import os
import json
import random
import re
import sys
import unicodedata
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


def normalize_word(text: str) -> str:
    """
    Normalizes a German word or phrase to its core form for comparison.
    Strips articles (der/die/das/ein/etc.), grammatical annotations like (m.), (f.), [v.],
    punctuation, and normalizes casing and whitespace.
    """
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    # Remove parenthetical / bracketed annotations e.g. (m.), (f.), (n.), (v.), [coll.]
    text = re.sub(r'\(.*?\)|\[.*?\]', '', text)
    # Remove leading definite/indefinite articles
    text = re.sub(
        r'^(?:der|die|das|den|dem|des|ein|eine|einer|einem|einen|eines)\s+',
        '',
        text.strip(),
        flags=re.IGNORECASE,
    )
    # Remove punctuation, keeping alphanumeric characters, hyphens, and whitespace
    text = re.sub(r'[^\w\s\-]', '', text, flags=re.UNICODE)
    # Normalize whitespace and convert to lowercase
    return " ".join(text.strip().lower().split())


# 1. Load previously learned words & deduplicate
HISTORY_FILE = "history.json"
raw_history = []
if os.path.exists(HISTORY_FILE):
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            raw_history = json.load(f)
    except Exception as e:
        print(f"Warning: Could not read {HISTORY_FILE}: {e}")
        raw_history = []

# Deduplicate history while preserving original order
history = []
history_normalized = set()
duplicates_found = False

for item in raw_history:
    norm = normalize_word(item)
    if norm and norm not in history_normalized:
        history_normalized.add(norm)
        history.append(item)
    else:
        duplicates_found = True

# Self-heal history.json if any duplicates were present
if duplicates_found:
    print(f"Cleaned up duplicate entries from {HISTORY_FILE}.")
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

print(f"Loaded {len(history)} unique historical words.")

# Curated thematic categories to ensure lexical diversity and avoid repetition
THEMES = [
    "Everyday life, home & daily routines (Alltag, Wohnen & Tagesablauf)",
    "Emotions, psychological states & mindset (Gefühle, Denkweisen & Psyche)",
    "Work, office, career & collaboration (Arbeitswelt, Büro & Beruf)",
    "Social interactions, human quirks & personality traits (Menschliche Eigenarten & Charakter)",
    "Nature, weather, seasons & environment (Natur, Jahreszeiten & Umwelt)",
    "Food, culinary traditions & dining culture (Essen, Trinken & Kulinarik)",
    "Clever, untranslatable compound nouns (Treffende deutsche Komposita)",
    "Colorful colloquial phrases & idioms (Bildhafte Redewendungen & Umgangssprache)",
    "Travel, mobility, city life & places (Unterwegs, Reisen, Stadt & Fortbewegung)",
    "Communication, humor, irony & social nuances (Kommunikation, Humor & Nuancen)",
]


def build_prompt(history_list: list[str], current_theme: str, session_rejected: set[str]) -> str:
    # Include up to the last 200 words (or all if smaller)
    forbidden_list = history_list[-200:] if len(history_list) > 200 else history_list
    forbidden_str = "\n".join(f"- {w}" for w in forbidden_list) if forbidden_list else "None yet"

    extra_rejected_warning = ""
    if session_rejected:
        rejected_str = ", ".join(f"'{w}'" for w in session_rejected)
        extra_rejected_warning = (
            f"\nATTENTION: You already attempted to suggest {rejected_str} during this session, "
            f"which was REJECTED as a duplicate. You MUST choose a completely different word!\n"
        )

    return f"""You are an expert German language tutor.
Your mission is to teach one interesting, practical German word or idiomatic phrase (Level B1-B2).

Today's focus area: {current_theme}

CRITICAL RULES:
1. ABSOLUTELY NO DUPLICATES: You must provide a brand-new word that is NOT in the forbidden list below.
2. Under NO circumstances choose any of these already taught words or their base words:
{forbidden_str}
{extra_rejected_warning}
3. Select an authentic, natural word or phrase that a native German speaker frequently uses. Avoid generic beginner clichés.

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


# 2. Call Gemini with programmatic duplicate validation and retry loop
client = genai.Client(api_key=gemini_api_key)

CANDIDATE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-4-argon",
]

MAX_ATTEMPTS = 6
data = None
session_rejected = set()
selected_theme = random.choice(THEMES)
print(f"Selected focus theme for today: '{selected_theme}'")

for attempt in range(1, MAX_ATTEMPTS + 1):
    model_name = CANDIDATE_MODELS[(attempt - 1) % len(CANDIDATE_MODELS)]
    print(f"\n[Attempt {attempt}/{MAX_ATTEMPTS}] Generating word with '{model_name}'...", flush=True)

    prompt = build_prompt(history, selected_theme, session_rejected)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.9,
                thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
            ),
        )
        candidate = json.loads(response.text)
    except Exception as e:
        print(f"Model '{model_name}' generation/parse failed: {e}", flush=True)
        continue

    # Validate essential schema keys
    required_keys = [
        "word_with_article",
        "part_of_speech",
        "plural",
        "english_translation",
        "example_german",
        "example_english",
        "memory_tip",
    ]
    if not all(k in candidate and str(candidate[k]).strip() for k in required_keys):
        print(f"Candidate JSON missing one or more required fields. Retrying...", flush=True)
        continue

    candidate_word = str(candidate["word_with_article"]).strip()
    norm_candidate = normalize_word(candidate_word)

    if not norm_candidate:
        print(f"Could not normalize candidate word '{candidate_word}'. Retrying...", flush=True)
        continue

    # HARD PROGRAMMATIC CHECK: Has this word (or its root) been taught before?
    if norm_candidate in history_normalized:
        print(
            f"DUPLICATE REJECTED: '{candidate_word}' (normalized: '{norm_candidate}') "
            f"already exists in history! Retrying...",
            flush=True,
        )
        session_rejected.add(candidate_word)
        # Shift theme on collision to steer away from the same semantic basin
        selected_theme = random.choice(THEMES)
        continue

    # Check plural form if provided
    plural_str = str(candidate.get("plural", "")).strip()
    norm_plural = normalize_word(plural_str)
    if (
        norm_plural
        and norm_plural not in {"kein plural", "none", "na"}
        and norm_plural in history_normalized
    ):
        print(
            f"DUPLICATE REJECTED: Plural '{plural_str}' (normalized: '{norm_plural}') "
            f"matches an existing word in history! Retrying...",
            flush=True,
        )
        session_rejected.add(candidate_word)
        selected_theme = random.choice(THEMES)
        continue

    # Candidate passed all checks
    data = candidate
    print(f"Approved unique word: '{candidate_word}' (normalized: '{norm_candidate}')", flush=True)
    break

if not data:
    sys.exit("Error: Failed to generate a unique, non-duplicate word after all attempts.")

# 3. Save new verified unique word to history
history.append(data["word_with_article"])
with open(HISTORY_FILE, "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

print(f"Successfully saved '{data['word_with_article']}' to {HISTORY_FILE}.")

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

print(f"Successfully posted '{data['word_with_article']}' to Discord!", flush=True)
