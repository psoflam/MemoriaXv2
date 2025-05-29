import spacy  # Import the spaCy library

nlp = spacy.load("en_core_web_sm")  # Load the small English model from spaCy

def analyze_input(user_input):
    doc = nlp(user_input)  # Process the user input to create a spaCy document
    entities = [(ent.text, ent.label_) for ent in doc.ents]  # Extract entities and their labels from the document
    return entities  # Return the list of entities and their labels
