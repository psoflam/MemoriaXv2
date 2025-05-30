from typing import List
import faiss
import numpy as np

class MemoryIndex:
    def __init__(self, embedding_dim: int = 384):  # Optional default value to fix tests
        print("Init called with dim:", embedding_dim)  # Optional debug
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.id_map = {}

    def add_memory(self, id: str, vector: List[float]):
        # Check dimensionality
        if len(vector) != self.index.d:
            raise ValueError(f"Vector dimensionality {len(vector)} does not match index dimensionality {self.index.d}")
        self.index.add(np.array([vector], dtype=np.float32))
        self.id_map[self.index.ntotal - 1] = id
        print(f"Memory added. Total count: {self.index.ntotal}")  # Debug print

    def query_similar(self, vector: List[float], top_k: int) -> List[str]:
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