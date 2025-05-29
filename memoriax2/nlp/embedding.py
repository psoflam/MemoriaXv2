from sentence_transformers import SentenceTransformer

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    """Generate an embedding for the given text."""
    return model.encode(text) 