from typing import List
import faiss
import numpy as np

class MemoryIndex:
    def __init__(self, embedding_dim: int = 384):  # Optional default value to fix tests
        print("Init called with dim:", embedding_dim)  # Optional debug
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.id_map = {}

    def add_memory(self, id: str, vector: List[float]):
        # Validate that the vector is a numpy array
        if not isinstance(vector, np.ndarray):
            print(f"[FAISS ERROR] Tried to add non-vector for key '{id}':", type(vector))
            return
        # Check dimensionality
        if len(vector) != self.index.d:
            raise ValueError(f"Vector dimensionality {len(vector)} does not match index dimensionality {self.index.d}")
        self.index.add(np.array([vector], dtype=np.float32))
        self.id_map[self.index.ntotal - 1] = id
        print(f"Memory added. Total count: {self.index.ntotal}")  # Debug print

    def query_similar(self, vector: List[float], top_k: int) -> List[str]:
        # Add a defensive type check
        if not isinstance(vector, np.ndarray):
            raise TypeError(f"Expected np.ndarray but got {type(vector)}")
        
        # Enforce dtype to be np.float32
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)
        
        D, I = self.index.search(np.array([vector], dtype=np.float32), top_k)
        return [self.id_map[i] for i in I[0] if i != -1]

    def rebuild_index(self):
        # Placeholder for batch update logic
        pass

    def reset_index(self):
        """Clear the index and id_map for fresh test runs."""
        self.index.reset()
        self.id_map.clear()
        print("Index and id_map have been reset.")  # Debug print 

    def list_keys(self):
        return list(self.id_map.keys())

    def get_top_similar_keys(self, query_vector, top_k=3):
        if not self.index.is_trained or self.index.ntotal == 0:
            return []
        
        query_vector = np.array([query_vector], dtype=np.float32)
        distances, indices = self.index.search(query_vector, top_k)

        # Map FAISS indices back to keys
        result_keys = []
        for idx in indices[0]:
            if idx in self.id_map:
                result_keys.append(self.id_map[idx])
        return result_keys

    def __len__(self):
        return self.index.ntotal

    def print_index_status(self):
        print(f"Total keys: {len(self.id_map)}")
        print(f"Index shape: ({self.index.ntotal}, {self.index.d})")

    def load_index_from_db(self, conn):
        cursor = conn.cursor()
        cursor.execute("SELECT key, embedding FROM memory_embeddings")
        rows = cursor.fetchall()

        for idx, (key, blob) in enumerate(rows):
            vec = np.frombuffer(blob, dtype=np.float32)
            self.add_memory(key, vec)

        print(f"[MemoryIndex] Loaded {len(rows)} items from DB into FAISS index.")

_memory_index: MemoryIndex | None = None


def get_memory_index(embedding_dim: int = 384) -> MemoryIndex:
    """Return the singleton MemoryIndex instance."""
    global _memory_index
    if _memory_index is None:
        _memory_index = MemoryIndex(embedding_dim)
    return _memory_index
