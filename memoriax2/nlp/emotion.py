import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def detect_emotion(text):
    lowered = text.lower()

    # Keyword override: basic emotion categories
    if any(word in lowered for word in ["angry", "furious", "mad", "rage", "infuriated"]):
        return "angry"
    if any(word in lowered for word in ["sad", "down", "depressed", "unhappy", "gloomy", "miserable"]):
        return "sad"
    if any(word in lowered for word in ["happy", "excited", "joyful", "ecstatic", "glad"]):
        return "happy"

    # Fallback to sentiment
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        return "happy"
    elif compound <= -0.05:
        return "sad"
    else:
        return "neutral" 