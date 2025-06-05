from sentence_transformers import SentenceTransformer
import numpy as np

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    """Generate an embedding for the given text."""
    vec = model.encode(text)
    assert isinstance(vec, np.ndarray), "embed_text did not return a numpy array!"
    assert vec.dtype == np.float32, "Embedding must be float32"
    assert vec.shape == (384,), f"Expected (384,) but got {vec.shape}"
    return vec 