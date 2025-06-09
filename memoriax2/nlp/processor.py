import spacy  # Import the spaCy library

nlp = spacy.load("en_core_web_sm")  # Load the small English model from spaCy


def analyze_input(user_input):
    """Analyze user input to extract entities and their labels."""
    doc = nlp(user_input)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities


def filter_input_for_embedding(user_input):
    """Filter user input to exclude certain phrases for embedding."""
    if user_input.lower().startswith("do you remember"):
        return ""
    return user_input
